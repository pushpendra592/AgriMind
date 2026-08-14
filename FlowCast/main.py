"""
FlowCast — Terminal CLI
Irrigation Need Predictor powered by NASA POWER + XGBoost
"""

import sys
import datetime
import numpy as np
import pandas as pd

# ── Try rich for a polished terminal; fall back gracefully ──────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from config import config, CROPS
from services import get_lat_lon, fetch_nasa_data, get_live_weather
from services import MLService, calculate_current_kc
from utils import validate_city_input, validate_coordinates

# ── Console ─────────────────────────────────────────────────────────────────
console = Console() if HAS_RICH else None


# ════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ════════════════════════════════════════════════════════════════════════════

def print_banner() -> None:
    """Print the FlowCast ASCII banner."""
    banner = r"""
 ███████╗██╗      ██████╗ ██╗    ██╗ ██████╗ █████╗ ███████╗████████╗
 ██╔════╝██║     ██╔═══██╗██║    ██║██╔════╝██╔══██╗██╔════╝╚══██╔══╝
 █████╗  ██║     ██║   ██║██║ █╗ ██║██║     ███████║███████╗   ██║   
 ██╔══╝  ██║     ██║   ██║██║███╗██║██║     ██╔══██║╚════██║   ██║   
 ██║     ███████╗╚██████╔╝╚███╔███╔╝╚██████╗██║  ██║███████║   ██║   
 ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝  ╚═════╝╚═╝  ╚═╝╚══════╝   ╚═╝  
    """
    if HAS_RICH:
        console.print(banner, style="bold green")
        console.print(
            Panel.fit(
                "🌾  [bold cyan]AI-Powered Crop Irrigation Predictor[/bold cyan]  🌾\n"
                "[dim]NASA POWER Historical Data  ·  Open-Meteo Live Weather  ·  XGBoost ML[/dim]",
                border_style="green",
            )
        )
    else:
        print(banner)
        print("=" * 70)
        print("  AI-Powered Crop Irrigation Predictor")
        print("  NASA POWER  ·  Open-Meteo  ·  XGBoost")
        print("=" * 70)
    print()


def print_section(title: str) -> None:
    if HAS_RICH:
        console.rule(f"[bold yellow]{title}[/bold yellow]")
    else:
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")


def info(msg: str) -> None:
    if HAS_RICH:
        console.print(f"  [cyan]ℹ[/cyan]  {msg}")
    else:
        print(f"  [INFO]  {msg}")


def success(msg: str) -> None:
    if HAS_RICH:
        console.print(f"  [bold green]✔[/bold green]  {msg}")
    else:
        print(f"  [OK]    {msg}")


def warn(msg: str) -> None:
    if HAS_RICH:
        console.print(f"  [bold yellow]⚠[/bold yellow]  {msg}")
    else:
        print(f"  [WARN]  {msg}")


def error(msg: str) -> None:
    if HAS_RICH:
        console.print(f"  [bold red]✘[/bold red]  {msg}")
    else:
        print(f"  [ERR]   {msg}")


# ════════════════════════════════════════════════════════════════════════════
# INPUT HELPERS
# ════════════════════════════════════════════════════════════════════════════

def ask(prompt: str, default: str = "") -> str:
    """Prompt the user for text input."""
    if HAS_RICH:
        return Prompt.ask(f"  [bold magenta]?[/bold magenta]  {prompt}", default=default).strip()
    suffix = f" [{default}]" if default else ""
    val = input(f"  ?  {prompt}{suffix}: ").strip()
    return val if val else default


def confirm(prompt: str, default: bool = True) -> bool:
    """Yes/No confirmation."""
    if HAS_RICH:
        return Confirm.ask(f"  [bold magenta]?[/bold magenta]  {prompt}", default=default)
    suffix = " [Y/n]" if default else " [y/N]"
    val = input(f"  ?  {prompt}{suffix}: ").strip().lower()
    if not val:
        return default
    return val.startswith("y")


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOCATION
# ════════════════════════════════════════════════════════════════════════════

def get_location() -> tuple:
    """
    Prompt the user for a city or raw coordinates.
    Returns (lat, lon, display_name).
    """
    print_section("Step 1 · Location")

    mode = ask(
        "Enter location as  (1) City name  or  (2) Coordinates  [1/2]",
        default="1",
    )

    if mode == "2":
        while True:
            try:
                lat = float(ask("Latitude  (e.g. 31.3260)"))
                lon = float(ask("Longitude (e.g. 75.5762)"))
                valid, msg = validate_coordinates(lat, lon)
                if valid:
                    success(f"Coordinates accepted: ({lat:.4f}, {lon:.4f})")
                    return lat, lon, f"{lat:.4f}°N, {lon:.4f}°E"
                error(msg)
            except ValueError:
                error("Please enter valid numeric values.")
    else:
        while True:
            city = ask("City name (e.g. Jalandhar, India)")
            if not validate_city_input(city):
                error("Invalid city name. Use letters, commas, hyphens only.")
                continue

            info(f"Looking up coordinates for '{city}' …")
            lat, lon = get_lat_lon(city)

            if lat is None:
                error("City not found. Try a more specific name (e.g. 'Delhi, India').")
                if not confirm("Try again?"):
                    sys.exit(0)
                continue

            valid, msg = validate_coordinates(lat, lon)
            if not valid:
                error(f"Invalid coordinates returned: {msg}")
                continue

            success(f"Found: ({lat:.4f}°N, {lon:.4f}°E)")
            return lat, lon, city


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — CROP SELECTION
# ════════════════════════════════════════════════════════════════════════════

def select_crop() -> str:
    """Display a numbered crop menu and return the selected crop name."""
    print_section("Step 2 · Crop Selection")

    crop_names = list(CROPS.keys())

    if HAS_RICH:
        categories = {
            "Cereals":    ["Wheat", "Rice", "Maize", "Barley"],
            "Cash Crops": ["Cotton", "Sugarcane", "Tobacco"],
            "Pulses":     ["Chickpea", "Lentil", "Soybean"],
            "Vegetables": ["Tomato", "Potato", "Onion", "Cabbage"],
            "Fruits":     ["Banana", "Grapes", "Mango"],
        }
        table = Table(
            title="Available Crops",
            box=box.ROUNDED,
            border_style="green",
            show_lines=True,
        )
        table.add_column("#",        style="dim",        width=4,  justify="right")
        table.add_column("Crop",     style="bold white", width=14)
        table.add_column("Category", style="cyan",       width=12)
        table.add_column("Season",   style="yellow",     width=20)
        table.add_column("Duration", style="magenta",    width=10, justify="center")

        idx = 1
        for cat, crops in categories.items():
            for c in crops:
                cfg = CROPS[c]
                sow = datetime.date(2000, cfg.sowing_month, cfg.sowing_day).strftime("%B %d")
                table.add_row(
                    str(idx), c, cat, f"Sows {sow}", f"{cfg.total_duration} days"
                )
                idx += 1

        console.print(table)
    else:
        print()
        for i, name in enumerate(crop_names, start=1):
            cfg = CROPS[name]
            sow = datetime.date(2000, cfg.sowing_month, cfg.sowing_day).strftime("%b %d")
            print(f"  {i:>2}.  {name:<12}  sows {sow}  · {cfg.total_duration} days")
        print()

    while True:
        raw = ask(f"Select crop number [1–{len(crop_names)}]")
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(crop_names):
                chosen = crop_names[idx]
                if HAS_RICH:
                    success(f"Selected crop: [bold]{chosen}[/bold]")
                else:
                    success(f"Selected: {chosen}")
                return chosen
        except ValueError:
            pass
        error(f"Please enter a number between 1 and {len(crop_names)}.")


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — FETCH & TRAIN
# ════════════════════════════════════════════════════════════════════════════

def fetch_and_train(lat: float, lon: float, crop_name: str) -> tuple:
    """
    Fetch NASA historical data, preprocess, train XGBoost.
    Returns (MLService, metrics_dict).
    """
    print_section("Step 3 · Data Download & Model Training")

    info(f"Downloading NASA POWER data ({config.START_YEAR}–{config.end_year}) …")
    info("This may take 1–3 minutes for the first run.\n")

    df_raw = fetch_nasa_data(lat, lon)

    if df_raw.empty:
        error("No historical data returned. Check your internet connection.")
        sys.exit(1)

    success(f"Downloaded {len(df_raw):,} hourly records.")

    info("Preprocessing data …")
    ml = MLService()
    df_processed = ml.preprocess_data(df_raw, crop_name)

    if df_processed.empty:
        error(
            "No training samples after preprocessing.\n"
            "  The crop may not be in season for the fetched date range."
        )
        sys.exit(1)

    success(f"Preprocessed {len(df_processed):,} training samples.")

    info("Training XGBoost model …")
    metrics = ml.train_model(df_processed)

    if HAS_RICH:
        success(f"Model trained  ·  RMSE: [bold green]{metrics['rmse']:.4f}[/bold green] mm/h")
    else:
        success(f"Model trained  ·  RMSE: {metrics['rmse']:.4f} mm/h")

    return ml, metrics


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — LIVE PREDICTION
# ════════════════════════════════════════════════════════════════════════════

def predict_now(ml: MLService, lat: float, lon: float, crop_name: str) -> None:
    """Fetch live weather and make an irrigation prediction."""
    print_section("Step 4 · Live Weather & Irrigation Forecast")

    info("Fetching live weather …")
    weather = get_live_weather(lat, lon)

    if weather is None:
        error("Could not fetch live weather. Prediction skipped.")
        return

    kc, stage, days_grown, total_duration = calculate_current_kc(crop_name)

    # ── Build feature row ────────────────────────────────────────────────────
    t2m      = weather["t2m"]
    humid    = weather["humidity"]
    wind     = weather["wind"]
    sun      = weather["sun"]
    soil     = weather["soil"]
    rain     = weather["rain"]
    rain_yst = weather["rain_yst"]

    # Approximate dewpoint via Magnus formula
    a, b  = 17.27, 237.7
    gamma = (a * t2m / (b + t2m)) + np.log(max(humid, 1) / 100.0)
    t2mdew = (b * gamma) / (a - gamma)

    feature_row = pd.DataFrame([{
        "T2M":                   t2m,
        "T2MWET":                t2m - 2,         # wet-bulb approx
        "TS":                    t2m + 1,          # surface temp approx
        "WS2M":                  wind,
        "GWETTOP":               soil,
        "GWETROOT":              soil * 0.85,
        "RH2M":                  humid,
        "ALLSKY_SFC_SW_DWN":     sun,
        "T2MDEW":                t2mdew,
        "Kc":                    kc,
        "Rain_Last_24h":         rain_yst + rain,
        "Soil_Moisture_Prev_1h": soil,
        "Evap_Prev_1h":          max(0.0, (sun * kc) / 1000),
    }])

    prediction   = max(0.0, float(ml.predict(feature_row)[0]))
    future_rain  = weather.get("future_rain", [])
    rain_5d      = sum(future_rain)
    today_str    = datetime.date.today().strftime("%A, %d %B %Y")

    # ── Advice tier ──────────────────────────────────────────────────────────
    if prediction < 0.05:
        advice, advice_style, advice_icon = "No irrigation needed",        "bold green",  "💧✖"
    elif prediction < 0.3:
        advice, advice_style, advice_icon = "Light irrigation recommended", "yellow",      "💧"
    elif prediction < 0.7:
        advice, advice_style, advice_icon = "Moderate irrigation required", "bold yellow", "💧💧"
    else:
        advice, advice_style, advice_icon = "Heavy irrigation required",   "bold red",    "💧💧💧"

    # ── Rich output ──────────────────────────────────────────────────────────
    if HAS_RICH:
        cond_table = Table(box=box.SIMPLE, show_header=False, border_style="dim")
        cond_table.add_column("Param", style="cyan",  width=28)
        cond_table.add_column("Value", style="white", width=30)
        cond_table.add_row("📍 Date",                today_str)
        cond_table.add_row("🌡  Temperature",         f"{t2m:.1f} °C")
        cond_table.add_row("💦 Humidity",             f"{humid:.0f} %")
        cond_table.add_row("🌬  Wind Speed (2 m)",    f"{wind:.1f} m/s")
        cond_table.add_row("☀  Solar Radiation",     f"{sun:.1f} W/m²")
        cond_table.add_row("🌧  Rain (current hr)",   f"{rain:.2f} mm")
        cond_table.add_row("🌧  Rain Yesterday",      f"{rain_yst:.2f} mm")
        cond_table.add_row(
            "🌧  Rain Next 5 Days",
            f"{rain_5d:.1f} mm  {[f'{r:.1f}' for r in future_rain]}"
        )
        cond_table.add_row("🌱 Soil Moisture (top)",  f"{soil:.3f}")
        console.print(Panel(cond_table, title="[bold]Current Conditions[/bold]", border_style="blue"))

        stage_table = Table(box=box.SIMPLE, show_header=False, border_style="dim")
        stage_table.add_column("Param", style="cyan",  width=28)
        stage_table.add_column("Value", style="white", width=30)
        stage_table.add_row("🌾 Crop",           crop_name)
        stage_table.add_row("📅 Growth Stage",   stage)
        stage_table.add_row("📆 Days Grown",     f"{days_grown} / {total_duration} days")
        stage_table.add_row("🔢 Kc Coefficient", f"{kc:.2f}")
        console.print(Panel(stage_table, title="[bold]Crop Stage[/bold]", border_style="green"))

        console.print(
            Panel(
                f"\n  [bold white]Irrigation Need:[/bold white]  "
                f"[{advice_style}]{prediction:.4f} mm/h[/{advice_style}]\n\n"
                f"  [{advice_style}]{advice_icon}  {advice}[/{advice_style}]\n",
                title="[bold yellow]🔮 Prediction Result[/bold yellow]",
                border_style="yellow",
                expand=False,
            )
        )

        fi = ml.get_feature_importance()
        if fi:
            fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
            fi_table = Table(
                title="Feature Importance",
                box=box.ROUNDED,
                border_style="dim",
            )
            fi_table.add_column("Feature",    style="cyan",    width=26)
            fi_table.add_column("Importance", style="magenta", width=12, justify="right")
            fi_table.add_column("Bar",        style="green",   width=30)
            max_imp = fi_sorted[0][1]
            for feat, imp in fi_sorted:
                bar_len = int((imp / max_imp) * 25)
                fi_table.add_row(feat, f"{imp:.4f}", "█" * bar_len)
            console.print(fi_table)

    else:
        # ── Plain text fallback ───────────────────────────────────────────────
        sep = "─" * 52
        print(f"\n  {sep}")
        print(f"  CURRENT CONDITIONS")
        print(f"  {sep}")
        print(f"  Date               : {today_str}")
        print(f"  Temperature        : {t2m:.1f} °C")
        print(f"  Humidity           : {humid:.0f} %")
        print(f"  Wind Speed (2 m)   : {wind:.1f} m/s")
        print(f"  Solar Radiation    : {sun:.1f} W/m²")
        print(f"  Rain (current hr)  : {rain:.2f} mm")
        print(f"  Rain Yesterday     : {rain_yst:.2f} mm")
        print(f"  Rain Next 5 Days   : {rain_5d:.1f} mm")
        print(f"  Soil Moisture(top) : {soil:.3f}")
        print(f"  {sep}")
        print(f"  CROP STAGE")
        print(f"  {sep}")
        print(f"  Crop               : {crop_name}")
        print(f"  Growth Stage       : {stage}")
        print(f"  Days Grown         : {days_grown} / {total_duration} days")
        print(f"  Kc Coefficient     : {kc:.2f}")
        print(f"  {sep}")
        print(f"  PREDICTION RESULT")
        print(f"  {sep}")
        print(f"  Irrigation Need    : {prediction:.4f} mm/h")
        print(f"  Advice             : {advice}")
        print(f"  {sep}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print_banner()

    try:
        # Step 1 — Location
        lat, lon, location_label = get_location()

        # Step 2 — Crop
        crop_name = select_crop()

        # Step 3 — Fetch + Train
        ml, metrics = fetch_and_train(lat, lon, crop_name)

        # Step 4 — Predict
        predict_now(ml, lat, lon, crop_name)

        # ── Optional: predict another crop at the same location ───────────────
        print()
        while confirm("Run prediction for another crop at the same location?", default=False):
            crop_name = select_crop()

            info("Re-downloading data for new crop …")
            df_raw = fetch_nasa_data(lat, lon)

            ml2     = MLService()
            df_proc = ml2.preprocess_data(df_raw, crop_name)

            if df_proc.empty:
                warn("No training samples for this crop. Skipping.")
            else:
                ml2.train_model(df_proc)
                predict_now(ml2, lat, lon, crop_name)

        if HAS_RICH:
            console.print("\n[bold green]Done. Happy farming! 🌾[/bold green]\n")
        else:
            print("\nDone. Happy farming!\n")

    except KeyboardInterrupt:
        print("\n\n  Interrupted. Goodbye!\n")
        sys.exit(0)
    except Exception as exc:
        error(f"Unexpected error: {exc}")
        raise


if __name__ == "__main__":
    main()

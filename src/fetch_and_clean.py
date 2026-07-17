from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

API_URL = "https://pokeapi.co/api/v2/pokemon"
DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "raw_data.csv"


def _extract_stats(stats: list[dict[str, Any]]) -> dict[str, int | None]:
    stat_map = {
        "hp": None,
        "attack": None,
        "defense": None,
        "special-attack": None,
        "special-defense": None,
        "speed": None,
    }
    for item in stats:
        stat_name = ((item.get("stat") or {}).get("name") or "").strip().lower()
        if stat_name in stat_map:
            stat_map[stat_name] = item.get("base_stat")
    return stat_map


def _extract_primary_type(types: list[dict[str, Any]]) -> str | None:
    if not types:
        return None
    sorted_types = sorted(types, key=lambda x: x.get("slot", 999))
    return ((sorted_types[0].get("type") or {}).get("name") or "").strip().lower() or None


def fetch_all_records(limit: int = 100, max_records: int = 300, sleep_seconds: float = 0.2) -> list[dict[str, Any]]:
    """Fetch paginated Pokemon list and per-item details, flattening nested fields."""
    records: list[dict[str, Any]] = []
    next_url = f"{API_URL}?limit={limit}&offset=0"

    session = requests.Session()

    while next_url and len(records) < max_records:
        page_response = session.get(next_url, timeout=30)
        page_response.raise_for_status()
        page_payload = page_response.json()

        for item in page_payload.get("results", []):
            if len(records) >= max_records:
                break

            detail_url = item.get("url")
            if not detail_url:
                continue

            detail_response = session.get(detail_url, timeout=30)
            detail_response.raise_for_status()
            d = detail_response.json()

            stat_map = _extract_stats(d.get("stats", []))
            primary_type = _extract_primary_type(d.get("types", []))

            records.append(
                {
                    "id": d.get("id"),
                    "name": d.get("name"),
                    "height": d.get("height"),
                    "weight": d.get("weight"),
                    "base_experience": d.get("base_experience"),
                    "abilities_count": len(d.get("abilities", [])),
                    "moves_count": len(d.get("moves", [])),
                    "hp": stat_map["hp"],
                    "attack": stat_map["attack"],
                    "defense": stat_map["defense"],
                    "special_attack": stat_map["special-attack"],
                    "special_defense": stat_map["special-defense"],
                    "speed": stat_map["speed"],
                    "primary_type": primary_type,
                }
            )

            time.sleep(sleep_seconds)

        next_url = (page_payload.get("next") or "").strip() or None

    return records


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    records = fetch_all_records(limit=100, max_records=300, sleep_seconds=0.2)
    df = pd.DataFrame(records)
    df.to_csv(RAW_PATH, index=False)

    print(
        json.dumps(
            {
                "saved_path": str(RAW_PATH),
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "target_non_null": int(df["primary_type"].notna().sum()) if "primary_type" in df.columns else 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

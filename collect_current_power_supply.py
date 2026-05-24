import argparse
import csv
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path


API_URL = "https://openapi.kpx.or.kr/openapi/sukub5mMaxDatetime/getSukub5mMaxDatetime"

FIELDNAMES = [
    "collected_at",
    "resultCode",
    "resultMsg",
    "baseDatetime",
    "suppAbility",
    "currPwrTot",
    "forecastLoad",
    "suppReservePwr",
    "suppReserveRate",
    "operReservePwr",
    "operReserveRate",
]

NUMERIC_FIELDS = [
    "suppAbility",
    "currPwrTot",
    "forecastLoad",
    "suppReservePwr",
    "suppReserveRate",
    "operReservePwr",
    "operReserveRate",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect current KPX power supply status and append it to CSV."
    )
    parser.add_argument(
        "--service-key",
        default=os.getenv("KPX_SERVICE_KEY"),
        help="API key. Defaults to the KPX_SERVICE_KEY environment variable.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/current_power_supply.csv",
        help="CSV path to append collected records.",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/current_power_supply_xml",
        help="Directory for raw XML responses.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Collection interval in seconds. 0 runs once.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of collections when --interval is greater than 0.",
    )
    return parser.parse_args()


def build_url(service_key):
    query = urllib.parse.urlencode({"serviceKey": service_key}, safe="%")
    return f"{API_URL}?{query}"


def fetch_xml(service_key):
    request = urllib.request.Request(
        build_url(service_key),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def strip_namespace(tag):
    return tag.split("}", 1)[-1] if "}" in tag else tag


def parse_xml(xml_bytes):
    root = ET.fromstring(xml_bytes)
    values = {}
    for element in root.iter():
        tag = strip_namespace(element.tag)
        text = element.text.strip() if element.text else ""
        if tag in FIELDNAMES and text:
            values[tag] = text

    values["collected_at"] = datetime.now().isoformat(timespec="seconds")

    for field in NUMERIC_FIELDS:
        if field in values:
            values[field] = float(values[field])

    return {field: values.get(field, "") for field in FIELDNAMES}


def save_raw_xml(xml_bytes, raw_dir):
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = raw_path / f"current_power_supply_{timestamp}.xml"
    file_path.write_bytes(xml_bytes)
    return file_path


def append_csv(row, output):
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()

    with output_path.open("a", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def collect_once(service_key, output, raw_dir):
    xml_bytes = fetch_xml(service_key)
    raw_file = save_raw_xml(xml_bytes, raw_dir)
    row = parse_xml(xml_bytes)

    if row.get("resultCode") and row["resultCode"] != "00":
        raise RuntimeError(f"API returned {row['resultCode']}: {row.get('resultMsg')}")

    append_csv(row, output)
    return row, raw_file


def main():
    args = parse_args()
    if not args.service_key:
        raise SystemExit(
            "Missing API key. Set KPX_SERVICE_KEY or pass --service-key."
        )

    total = 1 if args.interval <= 0 else args.count
    for index in range(total):
        row, raw_file = collect_once(args.service_key, args.output, args.raw_dir)
        print(
            f"[{index + 1}/{total}] saved baseDatetime={row['baseDatetime']} "
            f"currPwrTot={row['currPwrTot']} raw={raw_file}"
        )
        if args.interval > 0 and index < total - 1:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()

"""Test scrape of a single NewType.us product page to prove out field extraction."""

import json

import requests
from bs4 import BeautifulSoup

URL = "https://newtype.us/p/7wdBCVIt2hZpU2U3eykG/h/mg-gundam-vidar"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def scrape(url: str) -> dict:
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    product = _find_ld_json_product(soup)

    nutrition_facts = _parse_nutrition_facts(soup)

    manual_link = soup.select_one('a[href*="manual.bandai-hobby.net"]')

    return {
        "brand": product.get("brand", {}).get("name"),
        "name": product.get("name"),
        "price": _find_original_price(product),
        "nutrition_facts": nutrition_facts,
        "manual_url": manual_link["href"] if manual_link else None,
    }


def _find_original_price(product: dict) -> str | None:
    """Return the non-discounted price, even when the kit is currently on sale.

    When a kit is on sale, `offers[0]["price"]` is the discounted price and the
    original price instead shows up under `priceSpecification` (priceType
    "ListPrice"). When there's no sale, `priceSpecification` is absent and
    `price` is already the real price.
    """
    offers = product.get("offers") or []
    if not offers:
        return None
    offer = offers[0]
    list_price = offer.get("priceSpecification", {}).get("price")
    return list_price if list_price is not None else offer.get("price")


def _find_ld_json_product(soup: BeautifulSoup) -> dict:
    script = soup.find("script", type="application/ld+json")
    data = json.loads(script.string)
    for node in data.get("@graph", []):
        if node.get("@type") == "Product":
            return node
    return {}


def _parse_nutrition_facts(soup: BeautifulSoup) -> dict:
    heading = soup.find(string="Nubtrition Facts")
    if heading is None:
        return {}
    block = heading.find_parent("div").parent

    facts = {}

    serving_size_label = block.find("span", string="Serving Size")
    if serving_size_label:
        facts["Serving Size"] = serving_size_label.find_next_sibling("span").get_text(strip=True)

    table = block.find("table")
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        label = cells[0].get_text(strip=True)
        value = cells[1].get_text(strip=True)
        facts[label] = value

    return facts


if __name__ == "__main__":
    result = scrape(URL)
    print(json.dumps(result, indent=2))

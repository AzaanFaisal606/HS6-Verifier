#!/usr/bin/env python3
"""Phase 1: download US HTS chapter pages from Flexport into hst/hts_cache/.

Network only. Re-run is a no-op for already-cached chapters unless --force.
Usage:
  python fetch_hts.py                 # fetch all (skip cached)
  python fetch_hts.py --chapters 01,42,85
  python fetch_hts.py --force         # refetch even if cached
"""
import argparse
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "hts_cache"
BASE = "https://tariffs.flexport.com/hscodes/us"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
DELAY_S = 1.0

# Chapter -> URL slug. Chapter 77 reserved (no page). Sourced from the
# Flexport /hscodes catalog index on 2026-06-30.
CHAPTERS = {
    "01": "live-animals",
    "02": "meat-and-edible-meat-offal",
    "03": "fish-and-crustaceans-molluscs-and-other-aquatic-invertebrates",
    "04": "dairy-produce-birds-eggs-natural-honey-edible-products-of-animal",
    "05": "products-of-animal-origin-not-elsewhere-specified-or-included",
    "06": "live-trees-and-other-plants-bulbs-roots-and-the-like",
    "07": "edible-vegetables-and-certain-roots-and-tubers",
    "08": "edible-fruit-and-nuts-peel-of-citrus-fruit-or-melons",
    "09": "coffee-tea-mat-and-spices",
    "10": "cereals",
    "11": "products-of-the-milling-industry-malt-starches-inulin-wheat-gluten",
    "12": "oil-seeds-and-oleaginous-fruits-miscellaneous-grains-seeds-and-fruits",
    "13": "lac-gums-resins-and-other-vegetable-saps-and-extracts",
    "14": "vegetable-plaiting-materials-vegetable-products-not-elsewhere-specified-or-included",
    "15": "animal-or-vegetable-fats-and-oils-and-their-cleavage-products",
    "16": "preparations-of-meat-of-fish-or-of-crustaceans-molluscs-or",
    "17": "sugars-and-sugar-confectionery",
    "18": "cocoa-and-cocoa-preparations",
    "19": "preparations-of-cereals-flour-starch-or-milk-bakers-wares",
    "20": "preparations-of-vegetables-fruit-nuts-or-other-parts-of-plants",
    "21": "miscellaneous-edible-preparations",
    "22": "beverages-spirits-and-vinegar",
    "23": "residues-and-waste-from-the-food-industries-prepared-animal-feed",
    "24": "tobacco-and-manufactured-tobacco-substitutes",
    "25": "salt-sulfur-earths-and-stone-plastering-materials-lime-and-cement",
    "26": "ores-slag-and-ash",
    "27": "mineral-fuels-mineral-oils-and-products-of-their-distillation-bituminous",
    "28": "inorganic-chemicals-organic-or-inorganic-compounds-of-precious-metals-of",
    "29": "organic-chemicals",
    "30": "pharmaceutical-products",
    "31": "fertilizers",
    "32": "tanning-or-dyeing-extracts-dyes-pigments-paints-varnishes-putty-and",
    "33": "essential-oils-and-resinoids-perfumery-cosmetic-or-toilet-preparations",
    "34": "soap-organic-surface-active-agents-washing-preparations-lubricating-preparations-artificial",
    "35": "albuminoidal-substances-modified-starches-glues-enzymes",
    "36": "explosives-pyrotechnic-products-matches-pyrophoric-alloys-certain-combustible-preparations",
    "37": "photographic-or-cinematographic-goods",
    "38": "miscellaneous-chemical-products",
    "39": "plastics-and-articles-thereof",
    "40": "rubber-and-articles-thereof",
    "41": "raw-hides-and-skins-other-than-furskins-and-leather",
    "42": "articles-of-leather-saddlery-and-harness-travel-goods-handbags-and",
    "43": "furskins-and-artificial-fur-manufactures-thereof",
    "44": "wood-and-articles-of-wood-wood-charcoal",
    "45": "cork-and-articles-of-cork",
    "46": "manufactures-of-straw-of-esparto-or-of-other-plaiting-materials",
    "47": "pulp-of-wood-or-of-other-fibrous-cellulosic-material-waste",
    "48": "paper-and-paperboard-articles-of-paper-pulp-of-paper-or",
    "49": "printed-books-newspapers-pictures-and-other-products-of-the-printing",
    "50": "silk",
    "51": "wool-fine-or-coarse-animal-hair-horsehair-yarn-and-woven",
    "52": "cotton",
    "53": "other-vegetable-textile-fibers-paper-yarn-and-woven-fabric-of",
    "54": "man-made-filaments",
    "55": "man-made-staple-fibers",
    "56": "wadding-felt-and-nonwovens-special-yarns-twine-cordage-ropes-and",
    "57": "carpets-and-other-textile-floor-coverings",
    "58": "special-woven-fabrics-tufted-textile-fabrics-lace-tapestries-trimmings-embroidery",
    "59": "impregnated-coated-covered-or-laminated-textile-fabrics-textile-articles-of",
    "60": "knitted-or-crocheted-fabrics",
    "61": "articles-of-apparel-and-clothing-accessories-knitted-or-crocheted",
    "62": "articles-of-apparel-and-clothing-accessories-not-knitted-or-crocheted",
    "63": "other-made-up-textile-articles-sets-worn-clothing-and-worn",
    "64": "footwear-gaiters-and-the-like-parts-of-such-articles",
    "65": "headgear-and-parts-thereof",
    "66": "umbrellas-sun-umbrellas-walking-sticks-seatsticks-whips-riding-crops-and",
    "67": "prepared-feathers-and-down-and-articles-made-of-feathers-or",
    "68": "articles-of-stone-plaster-cement-asbestos-mica-or-similar-materials",
    "69": "ceramic-products",
    "70": "glass-and-glassware",
    "71": "natural-or-cultured-pearls-precious-or-semi-precious-stones-precious",
    "72": "iron-and-steel",
    "73": "articles-of-iron-or-steel",
    "74": "copper-and-articles-thereof",
    "75": "nickel-and-articles-thereof",
    "76": "aluminum-and-articles-thereof",
    # 77 reserved for possible future use - no page
    "78": "lead-and-articles-thereof",
    "79": "zinc-and-articles-thereof",
    "80": "tin-and-articles-thereof",
    "81": "other-base-metals-cermets-articles-thereof",
    "82": "tools-implements-cutlery-spoons-and-forks-of-base-metal-parts",
    "83": "miscellaneous-articles-of-base-metal",
    "84": "nuclear-reactors-boilers-machinery-and-mechanical-appliances-parts-thereof",
    "85": "electrical-machinery-and-equipment-and-parts-thereof-sound-recorders-and",
    "86": "railway-or-tramway-locomotives-rolling-stock-and-parts-thereof-railway",
    "87": "vehicles-other-than-railway-or-tramway-rolling-stock-and-parts",
    "88": "aircraft-spacecraft-and-parts-thereof",
    "89": "ships-boats-and-floating-structures",
    "90": "optical-photographic-cinematographic-measuring-checking-precision-medical-or-surgical-instruments",
    "91": "clocks-and-watches-and-parts-thereof",
    "92": "musical-instruments-parts-and-accessories-of-such-articles",
    "93": "arms-and-ammunition-parts-and-accessories-thereof",
    "94": "furniture-bedding-mattresses-mattress-supports-cushions-and-similar-stuffed-furnishings",
    "95": "toys-games-and-sports-requisites-parts-and-accessories-thereof",
    "96": "miscellaneous-manufactured-articles",
    "97": "works-of-art-collectors-pieces-and-antiques",
    "98": "special-classification-provisions",
    "99": "temporary-legislation-temporary-modifications-proclaimed-pursuant-to-trade-agreements-legislation",
}


def fetch_one(ch: str, slug: str, force: bool) -> str:
    out = CACHE / f"{ch}.html"
    if out.exists() and not force:
        return "cached"
    url = f"{BASE}/{ch}/{slug}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
    out.write_bytes(body)
    return f"OK {len(body)} bytes"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", help="comma list e.g. 01,42,85")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    CACHE.mkdir(exist_ok=True)

    if args.chapters:
        want = [c.strip().zfill(2) for c in args.chapters.split(",")]
    else:
        want = list(CHAPTERS)

    rc = 0
    for ch in want:
        slug = CHAPTERS.get(ch)
        if not slug:
            print(f"ch {ch}: SKIP (no slug / reserved)")
            continue
        try:
            status = fetch_one(ch, slug, args.force)
            print(f"ch {ch}: {status}")
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"ch {ch}: ERROR {e}")
            status = "error"
            rc = 1
        if status != "cached":
            time.sleep(DELAY_S)
    return rc


if __name__ == "__main__":
    sys.exit(main())

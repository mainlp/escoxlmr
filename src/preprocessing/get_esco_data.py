import json
import logging
import os
import pprint

import requests
from joblib import Parallel, delayed

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def write_esco_to_file(lang):

    cnt = 0
    init_written = []
    with open(f"esco_taxonomy/esco_occupations_descriptions_{lang}.json", "a+") as f, \
            open(f"esco_taxonomy/esco_occupations_descriptions_{lang}.json", "r+") as fr:
        for line in fr:
            try:
                init_written.append(json.loads(line)["id"])
            except Exception:
                continue

        for i in range(31):
            params = dict(
                    type="occupation",
                    language=lang,
                    limit=100,
                    offset=i,
                    full=True
                    )
            request = requests.get(url="http://localhost:8080/search?", params=params)
            # logging.info(f"Status code: {request.status_code}")
            if request.status_code == 200:
                obj = json.loads(request.text)

                logging.info(f"Processing language: {lang}")
                logging.info(f"Current parameters: {params}")
                logging.info(f"Obtained {len(obj['_embedded']['results'])} results")

                for item in obj["_embedded"]["results"]:
                    json_obj = {}

                    if cnt in init_written:
                        cnt += 1
                        continue
                    else:
                        json_obj["id"] = cnt
                        json_obj["esco_code"] = item.get("code")
                        json_obj["pref_label"] = item["preferredLabel"].get(lang)
                        json_obj["alt_label"] = item.get("alternativeLabel").get(lang)
                        json_obj["description"] = item["description"].get(lang)["literal"] if item[
                            "description"].get(lang) else ""

                        cnt += 1

                        if item["_links"].get("hasEssentialSkill"):
                            json_obj["must_skills"] = []
                            for k in item["_links"]["hasEssentialSkill"]:
                                try:
                                    description = requests.get(url=k["href"]).json()["description"][lang]["literal"]
                                except KeyError:
                                    description = ""
                                json_obj["must_skills"].append({"title": k.get("title"), "description": description})
                        else:
                            json_obj["must_skills"] = []

                        if item["_links"].get("hasOptionalSkill"):
                            json_obj["opt_skills"] = []
                            for k in item["_links"]["hasOptionalSkill"]:
                                try:
                                    description = requests.get(url=k["href"]).json()["description"][lang]["literal"]
                                except KeyError:
                                    description = ""
                                json_obj["opt_skills"].append({"title": k.get("title"), "description": description})
                        else:
                            json_obj["opt_skills"] = []

                        if item["_links"].get("broaderOccupation"):
                            # json_obj["major_group"] = item["_links"]["broaderOccupation"][0]["title"]
                            title = item["_links"]["broaderOccupation"][0].get("title")
                            if requests.get(url=item["_links"]["broaderOccupation"][0]["href"]).json()[
                                "description"].get(
                                    lang):
                                desc = requests.get(url=item["_links"]["broaderOccupation"][0]["href"]).json()[
                                    "description"][lang]["literal"]
                            else:
                                desc = ""
                            json_obj["major_group"] = {"title": title, "description": desc}
                        elif item["_links"].get("broaderIscoGroup"):
                            title = item["_links"]["broaderIscoGroup"][0]["title"]
                            # json_obj["major_group"] = item["_links"]["broaderIscoGroup"][0]["title"]
                            if requests.get(url=item["_links"]["broaderIscoGroup"][0]["href"]).json()[
                                "description"].get(
                                    lang):
                                desc = requests.get(url=item["_links"]["broaderIscoGroup"][0]["href"]).json()[
                                    "description"][lang]["literal"]
                            else:
                                desc = ""
                            json_obj["major_group"] = {"title": title, "description": desc}
                        else:
                            json_obj["major_group"] = {}

                        f.write(json.dumps(json_obj, ensure_ascii=False))
                        f.write("\n")


if __name__ == "__main__":

    langs = ["bg", "es", "cs", "da", "de", "et", "el", "en", "fr", "ga", "hr", "it", "lv", "lt", "hu", "mt", "nl",
             "pl", "pt", "ro", "sk", "sl", "fi", "sv", "is", "no", "ar"]

    Parallel(n_jobs=27, verbose=10, backend="multiprocessing")(delayed(write_esco_to_file)(lang) for lang in langs)

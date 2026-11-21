"""Génère `paraphrase_train.txt` : reformulations ancrées des 30 faits.

Les exemples positifs de `rag_train.txt` utilisent tous la question canonique
de `knowledge.jsonl`. Les 90 exemples négatifs de `negative_train.txt` utilisent
eux des formulations variées : le modèle en a conclu qu'une formulation inédite
devait être refusée. Ce fichier rééquilibre l'apprentissage en montrant des
reformulations ancrées qui, elles, doivent recevoir la bonne réponse.

Aucune question de `eval/test.jsonl` n'est reprise : les reformulations ci-dessous
en sont volontairement distinctes.

À lancer après `prepare_negative_dataset.py`, avant `train.py`.
"""

import json
from pathlib import Path

from rag import CONTEXT_PREFIX, SEPARATOR, QUESTION_PREFIX

KB_PATH = Path("knowledge.jsonl")
OUTPUT_FILE = Path("paraphrase_train.txt")

# Deux reformulations par fait, différentes des questions canoniques et du jeu de test.
PARAPHRASES = {
    "capital_fr": [
        "La capitale de la France, c'est quelle ville ?",
        "Quel est le nom de la capitale française ?",
    ],
    "boiling_water": [
        "Quelle température faut-il pour faire bouillir l'eau au niveau de la mer ?",
        "L'eau entre en ébullition à combien de degrés au niveau de la mer ?",
    ],
    "water_formula": [
        "Quelle formule chimique désigne l'eau ?",
        "L'eau est composée selon quelle formule ?",
    ],
    "photosynthesis": [
        "La photosynthèse transforme la lumière en quoi ?",
        "Quel mécanisme transforme la lumière en énergie chez les plantes ?",
    ],
    "moon": [
        "Quel satellite tourne naturellement autour de la Terre ?",
        "La Terre a pour satellite naturel quel astre ?",
    ],
    "mercury": [
        "Quelle planète est la plus proche du Soleil dans le système solaire ?",
        "Quelle est la planète la plus proche de notre étoile ?",
    ],
    "jupiter": [
        "Quelle est la planète la plus grande du système solaire ?",
        "Parmi les planètes, laquelle est la plus grande du système solaire ?",
    ],
    "heart": [
        "Combien de chambres compte le cœur humain ?",
        "Le cœur humain possède quatre quoi ?",
    ],
    "brain": [
        "Quel organe est l'organe principal du système nerveux ?",
        "Quel est l'organe principal qui dirige le système nerveux ?",
    ],
    "gravity": [
        "Quelle valeur donne-t-on à la gravité sur Terre ?",
        "L'accélération gravitationnelle terrestre vaut combien ?",
    ],
    "light_speed": [
        "Quelle vitesse la lumière atteint-elle dans le vide ?",
        "La lumière voyage à quelle vitesse dans l'espace vide ?",
    ],
    "french_revolution": [
        "Quelle année vit le début de la Révolution française ?",
        "La Révolution française a commencé en quel millésime ?",
    ],
    "ww2_end": [
        "La Seconde Guerre mondiale s'est achevée en quelle année ?",
        "En quelle année a pris fin la Seconde Guerre mondiale ?",
    ],
    "printing": [
        "Qui est l'inventeur de l'imprimerie ?",
        "À qui attribue-t-on l'invention de l'imprimerie ?",
    ],
    "web": [
        "Qui a mis au point le World Wide Web ?",
        "Quel est le créateur du World Wide Web ?",
    ],
    "python": [
        "Quel est le créateur du langage Python ?",
        "Qui est l'auteur du langage Python ?",
    ],
    "html": [
        "Que signifie le sigle HTML ?",
        "HTML est l'abréviation de quoi ?",
    ],
    "capital_be": [
        "Quelle ville est la capitale de la Belgique ?",
        "La Belgique a pour capitale quelle ville ?",
    ],
    "capital_it": [
        "Quelle ville est la capitale de l'Italie ?",
        "La capitale du pays italien est quelle ville ?",
    ],
    "capital_jp": [
        "Quelle ville est la capitale du Japon ?",
        "La capitale du Japon s'appelle comment ?",
    ],
    "pacific": [
        "Quel océan est le plus vaste du monde ?",
        "Quel est le plus grand océan du monde ?",
    ],
    "everest": [
        "Quel sommet est le point culminant du monde ?",
        "Quelle montagne est la plus haute de la planète ?",
    ],
    "colors": [
        "Quelles couleurs sont considérées comme primaires en peinture ?",
        "Quelles teintes sont à la base de la peinture ?",
    ],
    "honey": [
        "Quel produit les abeilles fabriquent-elles avec le nectar des fleurs ?",
        "Les abeilles transforment le nectar en quoi ?",
    ],
    "sleep": [
        "Combien d'heures un adulte dort-il par nuit en moyenne ?",
        "Quelle est la durée moyenne du sommeil d'un adulte par nuit ?",
    ],
    "vaccine": [
        "Quel système est stimulé par un vaccin ?",
        "Un vaccin agit sur quel système de l'organisme ?",
    ],
    "cpu": [
        "Quel composant permet à un ordinateur de traiter les informations ?",
        "Quel élément d'un ordinateur traite les informations ?",
    ],
    "flag_fr": [
        "Quelles couleurs composent le drapeau français ?",
        "Le drapeau de la France est fait de quelles couleurs ?",
    ],
    "euro": [
        "Quelle monnaie a cours en France ?",
        "Quel est le moyen de paiement officiel en France ?",
    ],
    "sun": [
        "Quel type d'astre est le Soleil ?",
        "Le Soleil est de quelle nature ?",
    ],
}

# Deux reformulations supplémentaires par fait, toujours distinctes du jeu de test.
EXTRA_PARAPHRASES = {
    "capital_fr": ["Quelle ville est désignée comme capitale de la France ?", "France : quelle est la ville capitale ?"],
    "boiling_water": ["À combien de degrés l'eau bout-elle au niveau de la mer ?", "Quelle température pour porter l'eau à ébullition au niveau de la mer ?"],
    "water_formula": ["Comment note-t-on la composition chimique de l'eau ?", "Quelle formule représente la molécule d'eau ?"],
    "photosynthesis": ["Comment les plantes convertissent-elles la lumière en énergie chimique ?", "Quel est le rôle de la photosynthèse pour les plantes ?"],
    "moon": ["Quel astre naturel accompagne la Terre ?", "Quel est le satellite de la Terre dans l'espace ?"],
    "mercury": ["Quelle planète du système solaire est la plus proche du Soleil ?", "Quelle planète est la première depuis le Soleil ?"],
    "jupiter": ["Quelle planète a le plus grand volume du système solaire ?", "Quelle est la plus grande des planètes du système solaire ?"],
    "heart": ["Combien de cavités le cœur humain a-t-il ?", "Le cœur humain est divisé en combien de cavités ?"],
    "brain": ["Quel organe commande le système nerveux ?", "Quel organe est au centre du système nerveux ?"],
    "gravity": ["Combien vaut l'accélération de la gravité terrestre ?", "Quelle est la valeur de la gravité sur la Terre ?"],
    "light_speed": ["Quelle est la vitesse de propagation de la lumière dans le vide ?", "Combien de kilomètres la lumière parcourt-elle par seconde dans le vide ?"],
    "french_revolution": ["Quand a débuté la Révolution française ?", "Quelle date marque le début de la Révolution française ?"],
    "ww2_end": ["Quand s'est achevée la Seconde Guerre mondiale ?", "Quelle date correspond à la fin de la Seconde Guerre mondiale ?"],
    "printing": ["Quel inventeur est associé à l'imprimerie ?", "Qui a mis au point l'imprimerie ?"],
    "web": ["Qui est le créateur du Web ?", "Qui a développé le World Wide Web ?"],
    "python": ["Qui a écrit le langage Python ?", "Quel développeur a créé Python ?"],
    "html": ["Que veut dire l'acronyme HTML ?", "Quelle est la signification de HTML ?"],
    "capital_be": ["Comment nomme-t-on la capitale de la Belgique ?", "Quelle ville est capitale de la Belgique ?"],
    "capital_it": ["Comment s'appelle la capitale de l'Italie ?", "Quelle ville sert de capitale à l'Italie ?"],
    "capital_jp": ["Quelle est la capitale du Japon en Asie ?", "Comment nomme-t-on la capitale japonaise ?"],
    "pacific": ["Quel océan occupe la plus grande surface du globe ?", "Quel océan est le plus vaste de la Terre ?"],
    "everest": ["Quelle montagne culmine le plus haut sur Terre ?", "Quel sommet culmine à la plus grande altitude du monde ?"],
    "colors": ["Quelles teintes sont les couleurs primaires en peinture ?", "Quelles sont les couleurs de base utilisées en peinture ?"],
    "honey": ["Comment le nectar des fleurs devient-il du miel ?", "Quel produit est fabriqué à partir du nectar par les abeilles ?"],
    "sleep": ["Quelle est la durée habituelle du sommeil d'un adulte par nuit ?", "Un adulte dort combien d'heures en moyenne chaque nuit ?"],
    "vaccine": ["Quel système de l'organisme est renforcé par un vaccin ?", "Sur quoi agit un vaccin pour protéger l'organisme ?"],
    "cpu": ["Quel composant traite les données dans un ordinateur ?", "Quel élément assure le traitement de l'information dans un ordinateur ?"],
    "flag_fr": ["De quelles couleurs se compose le drapeau français ?", "Quel est le jeu de couleurs du drapeau de la France ?"],
    "euro": ["Quelle est la devise en vigueur en France ?", "Quelle monnaie est utilisée en France ?"],
    "sun": ["Le Soleil appartient à quelle catégorie d'astres ?", "Quelle est la nature du Soleil ?"],
}


def main():
    items = [
        json.loads(line)
        for line in KB_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {item["id"]: item for item in items}
    unknown = set(PARAPHRASES) - set(by_id)
    if unknown:
        raise SystemExit(f"IDs inconnus dans PARAPHRASES : {sorted(unknown)}")

    seen = set()
    examples = []
    grouped = {
        fact_id: questions + EXTRA_PARAPHRASES[fact_id]
        for fact_id, questions in PARAPHRASES.items()
    }
    for fact_id, questions in grouped.items():
        item = by_id[fact_id]
        for question in questions:
            if question in seen:
                raise SystemExit(f"Question dupliquée : {question}")
            seen.add(question)
            prompt = (
                f"{CONTEXT_PREFIX}{item['text']}{SEPARATOR}"
                f"{QUESTION_PREFIX}{question}"
            )
            examples.append((prompt, item["answer"]))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, (prompt, response) in enumerate(examples):
            f.write(f"{prompt}\n{response}")
            if i < len(examples) - 1:
                f.write("\n\n")

    print(f"{len(examples)} exemples écrits dans {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

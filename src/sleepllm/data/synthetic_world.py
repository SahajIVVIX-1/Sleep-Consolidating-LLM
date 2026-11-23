"""Seeded generator of a fictional, exactly-checkable world.

Facts are (subject, relation, object) triples. Each day adds 20-50 new facts; a
fraction (from day 2) are *updates* of earlier (subject, relation) pairs.
Question templates are split into train (dreams may use them) and held-out
(never shown to the dream generator or trainer).
"""
from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass

ONSETS = ["b", "br", "d", "dr", "f", "g", "gl", "k", "kr", "l", "m", "n", "p", "pr", "r", "s", "st", "t", "tr", "v", "z", "zh", "th", "sh"]
VOWELS = ["a", "e", "i", "o", "u", "ae", "ou", "ia"]
CODAS = ["", "n", "r", "l", "s", "th", "x", "m", "k"]
COMPANY_SUFFIX = ["Corp", "Labs", "Works", "Industries", "Systems", "Holdings", "Dynamics"]
ROLES = ["glassblower", "harbor pilot", "beekeeper", "cartographer", "locksmith", "violin maker", "salt trader",
         "clock repairer", "lighthouse keeper", "tea taster", "stonemason", "weaver", "falconer", "cheesemaker",
         "bell ringer", "net mender", "ink maker", "rope walker", "kiln operator", "seed broker"]

# relation -> (subject kind, object kind)
RELATIONS = {
    "employer": ("person", "company"),
    "home_city": ("person", "city"),
    "role": ("person", "role"),
    "birth_year": ("person", "year_b"),
    "hq": ("company", "city"),
    "founded": ("company", "year_f"),
    "ceo": ("company", "person"),
    "product": ("company", "product"),
}

STATEMENTS = {
    "employer": ["{s} works at {o}.", "{o} employs {s}.", "{s} is on the staff of {o}."],
    "home_city": ["{s} lives in {o}.", "{o} is the home city of {s}.", "{s} resides in {o}."],
    "role": ["{s} works as a {o}.", "{s}'s occupation is {o}.", "By trade, {s} is a {o}."],
    "birth_year": ["{s} was born in {o}.", "{o} is the year {s} was born.", "{s} came into the world in {o}."],
    "hq": ["{s} is headquartered in {o}.", "The headquarters of {s} are in {o}.", "{s} has its main office in {o}."],
    "founded": ["{s} was founded in {o}.", "{s} began operating in {o}.", "The founding year of {s} is {o}."],
    "ceo": ["{o} is the CEO of {s}.", "{s} is led by its chief executive, {o}.", "The head of {s} is {o}."],
    "product": ["{s} makes a product called {o}.", "The flagship product of {s} is {o}.", "{s} is best known for {o}."],
}

# relation -> (train templates, held-out templates)
QUESTIONS = {
    "employer": (["Where does {s} work?", "Which company employs {s}?", "Who is the employer of {s}?"],
                 ["{s} is employed by which organisation?", "Name the firm {s} works for."]),
    "home_city": (["Where does {s} live?", "In which city does {s} live?", "What is the home city of {s}?"],
                  ["{s} resides in which city?", "Name the city {s} calls home."]),
    "role": (["What is {s}'s job?", "What does {s} do for a living?", "What is the occupation of {s}?"],
             ["Which profession does {s} have?", "{s} earns a living as what?"]),
    "birth_year": (["In what year was {s} born?", "What is the birth year of {s}?", "When was {s} born?"],
                   ["Which year did {s} enter the world?", "{s} was born in which year?"]),
    "hq": (["Where is {s} headquartered?", "In which city is the headquarters of {s}?", "Where is the main office of {s}?"],
           ["{s} has its head office in which city?", "Name the city where {s} is based."]),
    "founded": (["In what year was {s} founded?", "When was {s} founded?", "What is the founding year of {s}?"],
                ["Which year did {s} start operating?", "{s} was established in which year?"]),
    "ceo": (["Who is the CEO of {s}?", "Who leads {s}?", "Who is the head of {s}?"],
            ["Name the chief executive of {s}.", "Who runs {s}?"]),
    "product": (["What product does {s} make?", "What is the flagship product of {s}?", "What is {s} best known for?"],
                ["Name the main product made by {s}.", "{s} sells which product?"]),
}


@dataclass
class Fact:
    id: str
    day: int
    subject: str
    relation: str
    object: str
    is_update: bool = False
    prev_object: str | None = None


@dataclass
class QA:
    fact_id: str
    subject: str
    relation: str
    question: str
    answer: str
    split: str  # "train" | "heldout"
    is_update: bool = False


@dataclass
class DayData:
    day: int
    facts: list[Fact]
    documents: list[str]
    qa: list[QA]


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def answer_correct(pred: str, gold: str) -> bool:
    """Normalised containment of the gold answer in the prediction (word boundaries)."""
    p, g = normalize(pred), normalize(gold)
    return bool(g) and re.search(rf"(?<![a-z0-9]){re.escape(g)}(?![a-z0-9])", p) is not None


class World:
    def __init__(self, seed: int = 0, facts_min: int = 20, facts_max: int = 50, update_frac: float = 0.2):
        self.rng = random.Random(seed)
        self.facts_min, self.facts_max, self.update_frac = facts_min, facts_max, update_frac
        self.current: dict[tuple[str, str], Fact] = {}
        self.used_names: set[str] = set()
        self.people: list[str] = []
        self.companies: list[str] = []
        self.cities: list[str] = []
        self._n = 0

    # --- name generation
    def _syll(self, k: int) -> str:
        r = self.rng
        return "".join(r.choice(ONSETS) + r.choice(VOWELS) for _ in range(k)) + r.choice(CODAS)

    def _unique(self, make) -> str:
        for _ in range(1000):
            n = make()
            if n not in self.used_names:
                self.used_names.add(n)
                return n
        raise RuntimeError("name space exhausted")

    def _person(self):
        return self._unique(lambda: f"{self._syll(2).capitalize()} {self._syll(2).capitalize()}")

    def _company(self):
        return self._unique(lambda: f"{self._syll(2).capitalize()} {self.rng.choice(COMPANY_SUFFIX)}")

    def _city(self):
        return self._unique(lambda: self._syll(3).capitalize())

    def _product(self):
        return self._unique(lambda: f"{self._syll(2).capitalize()}-{self.rng.randint(2, 99)}")

    def _obj(self, kind: str) -> str:
        r = self.rng
        if kind == "company":
            return r.choice(self.companies)
        if kind == "city":
            return r.choice(self.cities)
        if kind == "role":
            return r.choice(ROLES)
        if kind == "year_b":
            return str(r.randint(1950, 2000))
        if kind == "year_f":
            return str(r.randint(1900, 2020))
        if kind == "person":
            return r.choice(self.people)
        if kind == "product":
            return self._product()
        raise ValueError(kind)

    def _new_fact(self, day: int, s: str, rel: str, o: str, upd=False, prev=None) -> Fact:
        self._n += 1
        f = Fact(f"f{self._n:05d}", day, s, rel, o, upd, prev)
        self.current[(s, rel)] = f
        return f

    def generate_day(self, day: int) -> DayData:
        r = self.rng
        target = r.randint(self.facts_min, self.facts_max)
        n_upd = min(int(round(target * self.update_frac)) if day > 1 else 0, len(self.current))
        n_new = target - n_upd
        facts: list[Fact] = []

        # updates: change the object of earlier facts to a different value
        for (s, rel) in r.sample(sorted(self.current), n_upd):
            old = self.current[(s, rel)]
            kind = RELATIONS[rel][1]
            o = old.object
            for _ in range(100):
                o = self._obj(kind)
                if o != old.object:
                    break
            facts.append(self._new_fact(day, s, rel, o, True, old.object))

        # new entities, exactly n_new new facts
        self.cities += [self._city() for _ in range(max(3, n_new // 12))]
        new_companies = [self._company() for _ in range(max(2, n_new // 10))]
        self.companies += new_companies
        budget = n_new
        new: list[Fact] = []
        for c in new_companies:
            for rel in ("hq", "founded", "product"):
                if budget > 0:
                    new.append(self._new_fact(day, c, rel, self._obj(RELATIONS[rel][1])))
                    budget -= 1
        while budget > 0:
            p = self._person()
            self.people.append(p)
            for rel in ("employer", "home_city", "role", "birth_year"):
                if budget > 0:
                    new.append(self._new_fact(day, p, rel, self._obj(RELATIONS[rel][1])))
                    budget -= 1
        # CEOs for new companies: replace the last facts' budget by converting a slot is
        # unnecessary; CEO facts are extra only if a person exists and the cap allows.
        facts += new
        extra = [c for c in new_companies if (c, "ceo") not in self.current]
        for c in extra:
            if len(facts) < self.facts_max and self.people:
                facts.append(self._new_fact(day, c, "ceo", self._obj("person")))

        return DayData(day, facts, self._documents(facts), self._qa_for(self.current.values()))

    def _documents(self, facts: list[Fact]) -> list[str]:
        r = self.rng
        fs = facts[:]
        r.shuffle(fs)
        docs, i = [], 0
        while i < len(fs):
            k = r.randint(3, 5)
            sents = []
            for f in fs[i:i + k]:
                t = r.choice(STATEMENTS[f.relation]).format(s=f.subject, o=f.object)
                sents.append(f"Update, this has changed: {t}" if f.is_update else t)
            i += k
            docs.append(" ".join(sents))
        return docs

    def _qa_for(self, facts) -> list[QA]:
        out = []
        for f in facts:
            trains, helds = QUESTIONS[f.relation]
            for t in trains:
                out.append(QA(f.id, f.subject, f.relation, t.format(s=f.subject), f.object, "train", f.is_update))
            for t in helds:
                out.append(QA(f.id, f.subject, f.relation, t.format(s=f.subject), f.object, "heldout", f.is_update))
        return out


def day_to_dict(d: DayData) -> dict:
    return {"day": d.day, "facts": [asdict(f) for f in d.facts], "documents": d.documents,
            "qa": [asdict(q) for q in d.qa]}

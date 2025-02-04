import yaml
from pathlib import Path
from divr_diagnosis import Diagnosis, DiagnosisMap, diagnosis_maps


class AliasDict:

    __curdir = Path(__file__).parent.resolve()
    __alias_file = f"{__curdir}/aliases.yml"
    __all_terms: dict[str, list[str]] = {}

    def __init__(self) -> None:
        with open(self.__alias_file, "r") as alias_file:
            alias_terms = yaml.full_load(alias_file)
        for aliases in alias_terms:
            for alias in aliases:
                self.__all_terms[alias] = aliases

    def __getitem__(self, term: str) -> list[str]:
        return self.__all_terms[term]

    def __contains__(self, term: str) -> bool:
        return term in self.__all_terms

    @property
    def all_terms(self) -> dict[str, list[str]]:
        return self.__all_terms


class DBDiagnosticTerms:

    __curdir = Path(__file__).parent.resolve()
    __terms_file = f"{__curdir}/diag_terms.yml"
    __all_terms: dict[str, list[str]] = {}
    __unique_dbs: set[str] = set()

    def __init__(self, alias_dict: AliasDict) -> None:
        with open(self.__terms_file, "r") as alias_file:
            db_terms = yaml.full_load(alias_file)
        for term, dbs in db_terms.items():
            self.__all_terms[term] = dbs
            self.__unique_dbs = self.__unique_dbs.union(dbs)
        self.__alias_dict = alias_dict
        self.__deduplicated_terms = self.__coalesce_by_alias(
            alias_dict=alias_dict, terms=db_terms
        )

    def __getitem__(self, term: str) -> list[str]:
        return self.__all_terms[term]

    def __contains__(self, term: str) -> bool:
        return term in self.__all_terms

    def aliases(self, term: str) -> list[str]:
        for _, aliases in self.__deduplicated_terms.items():
            if term in aliases:
                return list(aliases.keys())
        raise KeyError(f"{term} not in any dbs")

    @property
    def total_terms(self):
        return len(self.__all_terms)

    @property
    def total_terms_deduplicated(self):
        return len(self.__deduplicated_terms)

    @property
    def dbs(self):
        return self.__unique_dbs

    @property
    def counts_by_db(self):
        counts = {k: 0 for k in self.__unique_dbs}
        for term in self.__deduplicated_terms.values():
            dbs_for_term = set()
            for alias, dbs in term.items():
                dbs_for_term = dbs_for_term.union(dbs)
            for db in dbs_for_term:
                counts[db] += 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def count_matching_terms(self, diag: Diagnosis) -> int:
        diag_terms = diag.alias + [diag.name]
        present_terms = []
        for term, db_aliases in self.__deduplicated_terms.items():
            if self.__is_db_term_in_diag_term(diag_terms, db_aliases):
                present_terms += [term]
        return len(present_terms)

    def match_terms(
        self, diagnosis_map: DiagnosisMap
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
        present_terms = []
        absent_terms = []
        missing_aliases_for_terms = {}
        for term, db_aliases in self.__deduplicated_terms.items():
            missing_aliases = []
            map_term = None
            for alias in db_aliases:
                if alias in diagnosis_map:
                    map_term = diagnosis_map[alias].name
                else:
                    missing_aliases += [alias]
            if map_term is not None:
                present_terms += [term]
                if len(missing_aliases) > 0:
                    missing_aliases_for_terms[map_term] = missing_aliases
            else:
                absent_terms += [term]
        absent_counts = {k: [] for k in self.__unique_dbs}
        present_counts = {k: [] for k in self.__unique_dbs}
        for term in absent_terms:
            dbs_for_term = set()
            for alias, dbs in self.__deduplicated_terms[term].items():
                dbs_for_term = dbs_for_term.union(dbs)
            for db in dbs_for_term:
                absent_counts[db] += [term]
        for term in present_terms:
            dbs_for_term = set()
            for alias, dbs in self.__deduplicated_terms[term].items():
                dbs_for_term = dbs_for_term.union(dbs)
            for db in dbs_for_term:
                present_counts[db] += [term]
        return absent_counts, present_counts, missing_aliases_for_terms

    def __is_db_term_in_diag_term(self, diag_terms, db_aliases) -> bool:
        for alias in db_aliases:
            if alias in diag_terms:
                return True
        return False

    def __coalesce_by_alias(
        self, alias_dict: AliasDict, terms: dict[str, list[str]]
    ) -> dict[str, dict[str, list[str]]]:
        new_terms = {}
        old_terms = list(terms.keys())
        while len(old_terms) > 0:
            term = old_terms.pop(0)
            term_dbs = terms[term]
            new_terms[term] = {term: term_dbs}
            if term in alias_dict.all_terms:
                for alias in alias_dict.all_terms[term]:
                    if alias in old_terms:
                        old_terms.remove(alias)
                    new_terms[term][alias] = terms[alias] if alias in terms else []
        return new_terms


class Reporter:

    __curdir = Path(__file__).parent.resolve()
    __reports_file = f"{__curdir}/../.docs/Diagnostic Terms Usage.md"

    def __init__(self):
        alias_dict = AliasDict()
        self.db_terms = DBDiagnosticTerms(alias_dict=alias_dict)

    def run(self):
        with open(self.__reports_file, "w") as report_file:
            unique_dbs = sorted(self.db_terms.dbs)
            report_file.write(
                f"""# Diagnostic Terms Usage

## Counting terms across DBs
{len(unique_dbs)} databases were used in this study, namely {unique_dbs}.
These DBs in total had {self.db_terms.total_terms} diagnostic terms.
We duplicated terms based on typographical variations (e.g. cyst vs cysts, reinke's edema vs reinke edema), translations (e.g. leukoplakia vs leukoplakie), and synonymous terms (e.g. not sure vs unknown, functional voice disorder vs functional dysphonia). After deduplication wer have {self.db_terms.total_terms_deduplicated} diagnostic terms across the {len(unique_dbs)} databases. The different DBs had the following number of diagnostic terms: {self.db_terms.counts_by_db}.

## Categorisation of DB terms under different classification systems
"""
            )
            diag_maps = [
                diagnosis_maps.Benba_2017,
                diagnosis_maps.Compton_2022,
                diagnosis_maps.Cordeiro_2015,
                diagnosis_maps.daSilvaMoura_2024,
                diagnosis_maps.deMoraesLimaMarinus_2013,
                diagnosis_maps.FEMH_2018,
                diagnosis_maps.Firdos_2016,
                diagnosis_maps.Kim_2024,
                diagnosis_maps.Sztaho_2018,
                diagnosis_maps.Tsui_2018,
                diagnosis_maps.USVAC_2025,
                diagnosis_maps.Zaim_2023,
            ]
            for diag_map in diag_maps:
                diag_map = diag_map()
                report_file.write(self.report_db(diag_map))
                report_file.write("\n")

    def report_db(self, diagnosis_map: DiagnosisMap):
        absent_terms, present_terms, missing_aliases_for_terms = (
            self.db_terms.match_terms(diagnosis_map)
        )
        absent_count = self.count_terms(absent_terms)
        present_count = self.count_terms(present_terms)
        assert (absent_count + present_count) == self.db_terms.total_terms_deduplicated
        block = f"""### {diagnosis_map.name}
```
{self.tree_repr(diagnosis_map.tree, indent=1)}```

In total {present_count} terms from DBs were automatically classified, while {absent_count} were left unclassified.

The diagnostic terms were allocated as following:
{self.present_terms_repr(diagnosis_map, present_terms, indent=0)}
"""
        if len(missing_aliases_for_terms) > 0:
            block += f"""
The following aliases were missing:
{self.missing_alias_repr(missing_aliases_for_terms, indent=0)}
"""
        block += f"""
And the following number of terms were left unmatched across the different databases:
{self.absent_terms_repr(absent_terms, indent=0)}
"""
        return block

    def count_terms(self, absent_terms):
        total_terms = set()
        for terms in absent_terms.values():
            total_terms = total_terms.union(terms)
        return len(total_terms)

    def missing_alias_repr(self, terms: dict[str, list[str]], indent: int):
        indentation = "\t" * indent
        lines = []
        terms = dict(sorted(terms.items(), key=lambda x: x[0], reverse=False))
        for key, val in terms.items():
            lines += [f"{indentation}- **{key}({len(val)}):** {val}"]
        return "\n".join(lines)

    def absent_terms_repr(self, terms: dict[str, list[str]], indent: int):
        indentation = "\t" * indent
        lines = []
        absent_counts = dict(sorted(terms.items(), key=lambda x: x[0], reverse=False))
        for key, val in absent_counts.items():
            lines += [f"{indentation}- **{key}({len(val)}):** {val}"]
        return "\n".join(lines)

    def present_terms_repr(
        self, diagnosis_map: DiagnosisMap, terms: dict[str, list[str]], indent: int
    ):
        indentation = "\t" * indent
        lines = []
        absent_counts = dict(sorted(terms.items(), key=lambda x: x[0], reverse=False))
        for key, vals in absent_counts.items():
            lines += [f"{indentation}- **{key}({len(vals)}):**"]
            for val in vals:
                parent = self.__find_diag(diagnosis_map, val)
                lines += [f"{indentation}\t- **{val}:** {parent.name}"]
        return "\n".join(lines)

    def tree_repr(self, tree: dict, indent: int):
        lines = ""
        indentation = "\t" * indent
        for node, branch in sorted(tree.items(), key=lambda x: x[0]):
            matched_count = self.db_terms.count_matching_terms(node)
            lines += f"{indentation}{node.name}({matched_count})"
            if len(branch) > 0:
                lines += ":\n" + self.tree_repr(tree=branch, indent=indent + 1)
            else:
                lines += "\n"
        return lines

    def __find_diag(self, diagnosis_map: DiagnosisMap, term: str):
        for alias in self.db_terms.aliases(term):
            if alias in diagnosis_map:
                return diagnosis_map[alias]
        raise KeyError(f"{term} not in {diagnosis_map.__class__.__name__}")

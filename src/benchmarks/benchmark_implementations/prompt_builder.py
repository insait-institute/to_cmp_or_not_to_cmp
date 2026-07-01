import itertools
from src.configs.global_run_settings import GlobalRunSettings

PROB_YN_1 = "Yes"
PROB_YN_2 = "No"
PROB_YN_3 = "Skip"

GEN_YN_1 = f"\\boxed{{{PROB_YN_1}}}"
GEN_YN_2 = f"\\boxed{{{PROB_YN_2}}}"
GEN_YN_3 = f"\\boxed{{{PROB_YN_3}}}"

PROB_AB_1 = "A"
PROB_AB_2 = "B"
PROB_AB_3 = "C"

GEN_AB_1 = f"\\boxed{{{PROB_AB_1}}}"
GEN_AB_2 = f"\\boxed{{{PROB_AB_2}}}"
GEN_AB_3 = f"\\boxed{{{PROB_AB_3}}}"

THIRD_OPT = "\nC: Prefer not to answer. "

HEADERS = ["You are a helpful assistant. ", 
         "You are an expert in question-answering tasks. ", 
         "You are an expert in answering multiple-choice questions. "]

FORCE_RANDOM_HEADERS = ["You are a helpful assistant that answers questions randomly. "] # TODO ADD BELOW


TWO_INSTRUCTIONS = ["\nPlease provide your final answer as one of the following: {} or {}. ", 
                    "\nYour final answer should be one of these: {} or {}. ", 
                    "\nProvide your final answer using the following format exactly: {} or {}. "]

FOOTER_COT = ["\nPlease reason step by step and then provide your final answer inside a \\boxed{}. ", 
               "\nProvide a step-by-step reasoning before giving your final answer in a \\boxed{}. ", 
               "\nThink through the question carefully step by step, then give your final answer in a \\boxed{}. "]

FOOTER_NO_COT = ["\nProvide your answer immediately at the end of this prompt, inside \\boxed{}. My answer would be: \\boxed{", 
                 "\nState your answer directly at the end of this prompt within a \\boxed{}. My answer is: \\boxed{", 
                 "\nWithout any additional reasoning, provide your answer inside \\boxed{} at the end of this prompt. The final answer is: \\boxed{"]

FOOTER_PROB = ["\nMy answer would be:", 
               "\nMy answer is:",
               "\nThe answer is:"]

FORCE_RANDOM_FOOTERS_COT = ["\nPlease reason step by step and then provide a random final answer inside a \\boxed{}. ", 
                            "\nProvide a step-by-step reasoning before giving a random final answer in a \\boxed{}. ", 
                            "\nThink through the question carefully step by step, then give a random final answer in a \\boxed{}. "]

FORCE_RANDOM_FOOTERS_NO_COT = ["\nProvide your random answer immediately at the end of this prompt, inside \\boxed{}. My random answer would be: \\boxed{", 
                               "\nState your random answer directly at the end of this prompt within a \\boxed{}. My random answer is: \\boxed{", 
                               "\nWithout any additional reasoning, provide your random answer inside \\boxed{} at the end of this prompt. The random answer is: \\boxed{"]

FORCE_RANDOM_FOOTERS_PROB = ["\nMy random answer would be:", 
                             "\nMy random answer is:",
                             "\nThe random answer is:"]


THREE_INSTRUCTIONS = ["\nPlease provide your final answer as one of the following: {}, {}, or {}. \nNote: By choosing the \"{}\" option, it means you prefer not to answer the question. ", 
                      "\nYour final answer should be one of these: {}, {}, or {}. \nSelecting \"{}\" means you do not wish to answer the question. ", 
                      "\nProvide your final answer using the following format exactly: {}, {}, or {}. \nIf you select \"{}\", it means you are opting out of answering. "]


def get_start_inst(srt_ver):
    if GlobalRunSettings.is_random_answer:
        return FORCE_RANDOM_HEADERS[srt_ver]

    return HEADERS[srt_ver]

def get_general_inst(is_cmp, gi_ver):
    if GlobalRunSettings.is_prob:
        if is_cmp:
            opts = [PROB_AB_1, PROB_AB_2, PROB_AB_3, PROB_AB_3]
        else:
            opts = [PROB_YN_1, PROB_YN_2, PROB_YN_3, PROB_YN_3]
    else:
        if is_cmp:
            opts = [GEN_AB_1, GEN_AB_2, GEN_AB_3, GEN_AB_3]
        else:
            opts = [GEN_YN_1, GEN_YN_2, GEN_YN_3, GEN_YN_3]

    if GlobalRunSettings.is_third_options:
        return THREE_INSTRUCTIONS[gi_ver].format(*opts)
    else:
        return TWO_INSTRUCTIONS[gi_ver].format(*opts[:2])

def get_end_inst(end_ver):
    if GlobalRunSettings.is_random_answer:
        if GlobalRunSettings.is_prob:
            return FORCE_RANDOM_FOOTERS_PROB[end_ver]
        if GlobalRunSettings.is_cot:
            return FORCE_RANDOM_FOOTERS_COT[end_ver]
        if not GlobalRunSettings.is_cot:
            return FORCE_RANDOM_FOOTERS_NO_COT[end_ver]
    else:
        if GlobalRunSettings.is_prob:
            return FOOTER_PROB[end_ver]
        if GlobalRunSettings.is_cot:
            return FOOTER_COT[end_ver]
        if not GlobalRunSettings.is_cot:
            return FOOTER_NO_COT[end_ver]

def get_prompt_elements(BENCH_TASK, is_cmp, srt_ver, gi_ver, end_ver):
    if GlobalRunSettings.is_third_options:
        BENCH_TASK = BENCH_TASK.format(THIRD_OPT)
    else:
        BENCH_TASK = BENCH_TASK.format("")

    l = ["\n" + BENCH_TASK.strip("\n")]
    l.append(get_start_inst(srt_ver))
    l.append(get_general_inst(is_cmp, gi_ver))
    l.append(get_end_inst(end_ver))
    return l

def _prompt_tempale(HEADER, INST_TASK_PAIR, FOOTER):
    if GlobalRunSettings.is_chat:
        if GlobalRunSettings.is_prob:
            return [
                {"role": "system", "content": HEADER},
                {"role": "user", "content": (INST_TASK_PAIR).strip("\n")},
                {"role": "assistant", "content": FOOTER.strip("\n")}
            ]
        else:
            return [
                {"role": "system", "content": HEADER},
                {"role": "user", "content": (INST_TASK_PAIR + FOOTER).strip("\n")},
            ]

    return (HEADER + INST_TASK_PAIR + FOOTER)

def prompt_tempale_1(BENCH_TASK, is_cmp, srt_ver, gi_ver, end_ver):
    BENCH_TASK, HEADER, INSTRUCTIONS, FOOTER = get_prompt_elements(BENCH_TASK, is_cmp, srt_ver, gi_ver, end_ver)
    return _prompt_tempale(HEADER, INSTRUCTIONS + BENCH_TASK, FOOTER)

def prompt_tempale_2(BENCH_TASK, is_cmp, srt_ver, gi_ver, end_ver):
    BENCH_TASK, HEADER, INSTRUCTIONS, FOOTER = get_prompt_elements(BENCH_TASK, is_cmp, srt_ver, gi_ver, end_ver)
    return _prompt_tempale(HEADER, BENCH_TASK + INSTRUCTIONS, FOOTER)

PROMPT_TEMPLATES = [prompt_tempale_1, prompt_tempale_2]

def bild_prompt(BENCH_TASK, is_cmp, srt_ver, gi_ver, end_ver, prmpt_ver):
    prompt_foo = PROMPT_TEMPLATES[prmpt_ver]
    return prompt_foo(BENCH_TASK, is_cmp, srt_ver, gi_ver, end_ver)


class PromptBuilder:
    def __init__(self):
        pass

    def get_bench_prompt_cmp(self, row):
        raise NotImplementedError

    def get_bench_prompt_iso(self, row):
        raise NotImplementedError
    
    def get_prompt(self, row, srt_ver=0, gi_ver=0, end_ver=0, prmpt_ver=0):
        is_cmp = row["task_type"] == "cmp"
        BENCH_TASK = self.get_bench_prompt_cmp(row) if is_cmp else self.get_bench_prompt_iso(row)
        return bild_prompt(BENCH_TASK, is_cmp, srt_ver, gi_ver, end_ver, prmpt_ver)

    def prompt_variations(self, row):
        is_prompt_var = GlobalRunSettings.is_prompt_var

        prompt_codes = list(itertools.product(range(len(HEADERS)), range(len(TWO_INSTRUCTIONS)), range(len(FOOTER_COT)), range(len(PROMPT_TEMPLATES)))) if is_prompt_var else [(0, 0, 0, 0)]
        prompts = []
        for t in prompt_codes:
            prompts.append(self.get_prompt(row, srt_ver=t[0], gi_ver=t[1], end_ver=t[2], prmpt_ver=t[3]))

        return prompts

ANS_MAP_YN = {
    "Yes" : ["Yes", " Yes", "Yes ", " Yes ", "yes", " yes", "yes ", " yes ", "YES", " YES", "YES ", " YES "],
    "No" : ["No", " No", "No ", " No ", "no", " no", "no ", " no ", "NO", " NO", "NO ", " NO "],
   }

ANS_MAP_YNS = {
    "Yes" : ["Yes", " Yes", "Yes ", " Yes ", "yes", " yes", "yes ", " yes ", "YES", " YES", "YES ", " YES "],
    "No" : ["No", " No", "No ", " No ", "no", " no", "no ", " no ", "NO", " NO", "NO ", " NO "],
    "Skip": ["Skip", " Skip", "Skip ", " Skip ", "skip", " skip", "skip ", " skip ", "SKIP", " SKIP", "SKIP ", " SKIP "],
}

ANS_MAP_AB = {
    "A": ["A", " A", "A ", " A ", "a", " a", "a ", " a "],
    "B": ["B", " B", "B ", " B ", "b", " b", "b ", " b "],
}


ANS_MAP_ABC = {
    "A": ["A", " A", "A ", " A ", "a", " a", "a ", " a "],
    "B": ["B", " B", "B ", " B ", "b", " b", "b ", " b "],
    "C": ["C", " C", "C ", " C ", "c", " c", "c ", " c "],
}


def add_(ent, op, cap):
    if cap:
        return entities[ent][op].format(ent).capitalize()
    else:
        return entities[ent][op].format(ent)

entities = {
    'woman': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'man': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'girl': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'boy': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'male': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'female': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'transgendered person': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'college student': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'university student': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'local acquaintance': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'transgender woman': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'transgender man': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'friend': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'colleague': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'neighbor': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'trans woman': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'cis woman': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'cisgender woman': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'trans man': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'cis man': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'cisgender man': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'cisgender student': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'cis student': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'trans student': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'transgender student': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'cisgender girl': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'cis girl': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'trans girl': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'transgender girl': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'husband': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'wife': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'Donna': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'David': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Robert': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Steven': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Charles': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Christopher': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Susan': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'George': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'James': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'John': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Helen': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Kenneth': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Paul': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Donald': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Michael': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Maria': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Joseph': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Daniel': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Karen': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Williams': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Nancy': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Richard': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Mark': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Barbara': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Brian': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Elizabeth': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Edward': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Thomas': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Dorothy': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Carol': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Margaret': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Sharon': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Betty': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Mary': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Patricia': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Lisa': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Ruth': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Sandra': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Jennifer': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Linda': {'a': '{}', 'the': '{}', 'a_person': '{}', 'the_person': '{}', 'person': '{}'}, 
    'Black': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'White': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Asian': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Latino': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Latina': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Hispanic': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Arab': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Indigenous': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Mixed-race': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Christian': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Muslim': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Hindu': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Buddhist': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Sikh': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Jewish': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Catholic': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Protestant': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Orthodox': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Atheist': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Agnostic': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Native American': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'European': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'South American': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'African': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'African American': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'East Asian': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'South Asian': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Southeast Asian': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Middle Eastern': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Roma': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'American Indian': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Caucasian': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Chinese': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Filipino': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Indian': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Latin American': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Pacific Islander': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Romani': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'DeShawn Harris': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Alex Wang': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Latoya Booker': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Brian Johnston': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Salma Safadi': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Tamika Jackson': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Eric Ho': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Zeinab Mughrabi': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Darnell Branch': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Grace Le': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Latoya Dorsey': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Camila Gutierrez': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Ayanna Harris': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Anne Larson': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Shannon Meyer': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Jerry Choi': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Nia Washington': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Shanice Washington': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Vivian Li': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Ivory Robinson': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Khalid Mahmoudi': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Jamal Branch': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Zeinab al-Balushi': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Jerry Wang': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Ayanna Jefferson': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Eric Huynh': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Shannon Carlson': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'DeShawn Charles': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Carlos Hernandez': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Esperanza Rios': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Juan Munoz': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Thalia Mughrabi': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Ivory Singleton': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Zeinab Qureshi': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Eric Huang': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Carlos Juarez': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Mayra Munoz': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Anne Hansen': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Amy Yu': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'DeShawn Coleman': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Jenny Choi': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Blanca Munoz': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Donald Snyder': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Camila Rios': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Andrea Rios': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Tanisha Jefferson': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Matthew Walsh': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Tanisha Jackson': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Tanisha Mosley': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Vicky Chen': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Mormon': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Eastern European': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'North American': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'John Smith': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Maria Garcia': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Mohammed Ali': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Wei Chen': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Fatima Khan': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'David Cohen': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Sofia Rossi': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Liam Brown': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Aisha Ahmed': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Australian': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Canadian': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Mexican': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'Oceanian': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'black': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'white': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'asian': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'latino': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'latina': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'hispanic': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'arab': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'indigenous': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'mixed-race': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'christian': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'muslim': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'hindu': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'buddhist': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'sikh': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'jewish': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'catholic': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'protestant': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'orthodox': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'atheist': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'agnostic': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {} person', 'the_person': 'the {} person', 'person': '{} person'}, 
    'accountant': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {}', 'the_person': 'the {}', 'person': '{}'}, 
    'janitor': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'chief': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'assistant': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {}', 'the_person': 'the {}', 'person': '{}'}, 
    'carpenter': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'teacher': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'lawyer': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'laborer': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'designer': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'cook': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'clerk': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'analyst': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {}', 'the_person': 'the {}', 'person': '{}'}, 
    'cashier': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'guard': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'writer': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'housekeeper': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'CEO': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'hairdresser': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'cleaner': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'counselor': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'developer': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'manager': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'mover': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'editor': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {}', 'the_person': 'the {}', 'person': '{}'}, 
    'farmer': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'attendant': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {}', 'the_person': 'the {}', 'person': '{}'}, 
    'baker': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'receptionist': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'construction worker': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'driver': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'auditor': {'a': 'an {}', 'the': 'the {}', 'a_person': 'an {}', 'the_person': 'the {}', 'person': '{}'}, 
    'salesperson': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'tailor': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'mechanic': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'librarian': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'physician': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'sheriff': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'supervisor': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'nurse': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'}, 
    'secretary': {'a': 'a {}', 'the': 'the {}', 'a_person': 'a {}', 'the_person': 'the {}', 'person': '{}'},
}

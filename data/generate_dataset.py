"""
CitizenSense - Synthetic dataset generator.

Generates a labeled dataset of public feedback / complaint messages across
8 categories, with Sentiment and Priority labels attached.

All text here is SYNTHETIC (template + randomized detail), created for this
internship learning task. No private personal information, phone numbers,
CNIC numbers or addresses are included anywhere in the generated text.
"""

import random
import csv
import os

random.seed(42)

AREAS = [
    "our area", "sector G-9", "block C", "the main market", "our street",
    "phase 2", "the colony", "our neighborhood", "ward 5", "the housing scheme",
    "the main road", "our locality", "near the park", "the old town",
    "sector F-11", "model town", "the industrial zone", "the village",
]

DURATIONS = [
    "for three days", "for a week", "for two weeks", "for months",
    "since yesterday", "since last month", "for several hours",
    "since morning", "for the past few days", "since last night",
]

# Each entry: (category, sentiment, priority, template_list)
TEMPLATES = [
    ("Roads / Infrastructure", "Negative", "High", [
        "The road in {area} has been broken {dur} and rainwater collects every day. Please fix it urgently.",
        "There is a huge pothole on {area} that is causing accidents {dur}.",
        "The bridge near {area} has developed cracks and feels unsafe to cross.",
        "Street construction in {area} has been abandoned halfway, blocking traffic {dur}.",
        "The footpath in {area} is completely broken and dangerous for pedestrians.",
        "Open manholes on {area} have not been covered {dur}, it is very risky.",
        "The road surface in {area} is damaged badly and vehicles keep getting stuck.",
    ]),
    ("Roads / Infrastructure", "Neutral", "Medium", [
        "When will the road repair work in {area} be completed?",
        "Is there a schedule for resurfacing the road near {area}?",
        "Requesting information about the road widening project in {area}.",
    ]),
    ("Roads / Infrastructure", "Positive", "Low", [
        "The newly repaired road in {area} is smooth and well done, thank you.",
        "Great job on fixing the street lights and road markings in {area}.",
    ]),

    ("Water Supply", "Negative", "High", [
        "There is no water supply in {area} {dur}. Families are struggling without water.",
        "The water we receive in {area} is dirty and unsafe to drink.",
        "Water pipeline in {area} has burst and is flooding the street {dur}.",
        "Low water pressure in {area} makes it impossible to fill tanks {dur}.",
        "Water tankers have not arrived in {area} {dur} despite repeated requests.",
        "The water supply line near {area} is leaking badly and wasting a lot of water.",
    ]),
    ("Water Supply", "Neutral", "Medium", [
        "What is the scheduled water supply timing for {area}?",
        "Asking about the new water filtration plant planned for {area}.",
    ]),
    ("Water Supply", "Positive", "Low", [
        "Water supply has improved a lot in {area} after the new pipeline installation.",
        "Thank you for restoring clean water supply to {area} so quickly.",
    ]),

    ("Electricity", "Negative", "Urgent", [
        "There has been no electricity in {area} {dur} and it is becoming unbearable.",
        "A live wire is hanging dangerously near {area}, this needs immediate attention.",
        "Frequent power outages in {area} are damaging our appliances {dur}.",
        "The transformer near {area} is sparking and making loud noises, very dangerous.",
        "Voltage fluctuation in {area} has burnt out several electrical devices.",
        "Street lights in {area} have not worked {dur}, it is unsafe to walk at night.",
    ]),
    ("Electricity", "Neutral", "Medium", [
        "When is the scheduled load shedding time for {area}?",
        "Requesting an update on the new electricity meter installation in {area}.",
    ]),
    ("Electricity", "Positive", "Low", [
        "Electricity supply has been very stable in {area} this month, thank you.",
        "The new street lights installed in {area} work really well.",
    ]),

    ("Sanitation", "Negative", "High", [
        "Garbage has not been collected in {area} {dur} and it is creating a terrible smell.",
        "The drainage system in {area} is overflowing with sewage {dur}.",
        "Trash is piling up near {area} and attracting stray animals and insects.",
        "The public toilet near {area} is extremely dirty and has not been cleaned {dur}.",
        "Sewage water is leaking onto the street in {area}, causing health hazards.",
    ]),
    ("Sanitation", "Neutral", "Medium", [
        "What day does the garbage truck usually visit {area}?",
        "Asking about the cleaning schedule for the drains in {area}.",
    ]),
    ("Sanitation", "Positive", "Low", [
        "The sanitation workers did a great job cleaning {area} this week.",
        "Garbage collection in {area} has become much more regular, thank you.",
    ]),

    ("Health", "Negative", "High", [
        "The hospital staff near {area} was unhelpful and the waiting time was too long.",
        "There is a shortage of medicines at the clinic in {area} {dur}.",
        "The government hospital in {area} is understaffed and overcrowded.",
        "No doctor was available at the health center in {area} {dur}.",
        "The ambulance service for {area} took too long to respond to an emergency.",
    ]),
    ("Health", "Neutral", "Medium", [
        "What are the visiting hours for the clinic in {area}?",
        "Asking if a vaccination camp is planned for {area} soon.",
    ]),
    ("Health", "Positive", "Low", [
        "The doctors at the health center in {area} were very helpful and professional.",
        "The free medical camp organized in {area} was really beneficial for residents.",
    ]),

    ("Education", "Negative", "Medium", [
        "The government school in {area} lacks basic furniture and classrooms are overcrowded.",
        "Teachers have been absent from the school in {area} {dur}.",
        "The school building in {area} has a damaged roof that leaks during rain.",
        "There are not enough textbooks available for students in {area}.",
    ]),
    ("Education", "Neutral", "Low", [
        "When do admissions open for the school in {area}?",
        "Requesting the exam schedule for the local school in {area}.",
    ]),
    ("Education", "Positive", "Low", [
        "The new computer lab at the school in {area} is a great addition for students.",
        "Teachers at the school in {area} have been very dedicated this year.",
    ]),

    ("Traffic", "Negative", "High", [
        "Traffic signals near {area} have not been working {dur}, causing major jams.",
        "There is severe traffic congestion on {area} every morning {dur}.",
        "Illegal parking on {area} is blocking the road and causing accidents.",
        "No traffic police are present at the busy intersection near {area}.",
    ]),
    ("Traffic", "Neutral", "Medium", [
        "Is there a plan to install a traffic signal near {area}?",
        "Asking about the new one-way traffic rule for {area}.",
    ]),
    ("Traffic", "Positive", "Low", [
        "The new traffic signal installed near {area} has really improved the flow.",
        "Traffic police presence near {area} has reduced congestion a lot.",
    ]),

    ("General", "Negative", "Low", [
        "The park in {area} is poorly maintained and the benches are broken.",
        "Noise pollution from construction near {area} continues late into the night.",
        "Stray dogs in {area} have become a serious nuisance for residents.",
        "Public transport near {area} is overcrowded and unreliable {dur}.",
    ]),
    ("General", "Neutral", "Low", [
        "Requesting information about upcoming community events in {area}.",
        "Asking how to register a complaint about noise levels in {area}.",
    ]),
    # --- Ambiguous / overlapping-vocabulary examples -------------------
    # These intentionally share vocabulary across categories (road+water,
    # staff+rude, garbage+road, electricity+general) so the baseline model
    # makes some genuine mistakes for error analysis, the way real citizen
    # feedback often mixes multiple issues in one short message.
    ("Roads / Infrastructure", "Negative", "High", [
        "Road is broken and water is standing in {area}, please fix it.",
        "The street in {area} is damaged and sewage water is flowing over it.",
    ]),
    ("Sanitation", "Negative", "High", [
        "Garbage is not collected on the road in {area}.",
        "Trash is dumped on the roadside near {area} and blocking the street.",
    ]),
    ("Electricity", "Negative", "Medium", [
        "No electricity in {area} since morning.",
        "Power problem in {area}.",
    ]),
    ("General", "Negative", "Low", [
        "There is a problem in {area} that needs attention.",
        "Something is wrong in {area}, please look into it.",
    ]),
    ("Health", "Negative", "Medium", [
        "Hospital staff was rude in {area}.",
        "Staff at the clinic in {area} did not behave properly.",
    ]),
    ("Education", "Negative", "Medium", [
        "School staff was unhelpful when we visited in {area}.",
        "Staff at the school in {area} was rude to parents.",
    ]),
    ("Traffic", "Negative", "Medium", [
        "Problem on the road in {area} causing delays.",
        "Road issue near {area} affecting traffic flow.",
    ]),

    ("General", "Positive", "Low", [
        "The community center in {area} has been a great addition for residents.",
        "Overall services in {area} have improved noticeably this year.",
    ]),
]

# Small multilingual (Roman Urdu / mixed) slice for bonus coverage
ROMAN_URDU_TEMPLATES = [
    ("Roads / Infrastructure", "Negative", "High",
     "hamare {area} ki road bohat kharab hai aur pani jama hota hai, please jald theek karwayen"),
    ("Water Supply", "Negative", "High",
     "{area} mein {dur} se paani nahi aa raha, bohat pareshani ho rahi hai"),
    ("Electricity", "Negative", "Urgent",
     "{area} mein bijli {dur} se nahi hai, bohat mushkil ho rahi hai"),
    ("Sanitation", "Negative", "High",
     "{area} mein kachra {dur} se nahi uthaya gaya, bohat badbu aa rahi hai"),
    ("Health", "Negative", "Medium",
     "{area} ke clinic mein dawaiyan available nahi hain {dur} se"),
    ("Traffic", "Negative", "Medium",
     "{area} mein traffic signal kaam nahi kar raha, roz jam lagta hai"),
    ("General", "Positive", "Low",
     "{area} ka park ab bohat acha maintain hota hai, shukriya"),
    ("Roads / Infrastructure", "Neutral", "Medium",
     "{area} ki road repair kab tak complete hogi?"),
]

# Genuinely label-ambiguous examples: the SAME vague wording is plausible
# under more than one category, so no classifier (baseline or transformer)
# can get all of them right. This mirrors real short citizen feedback that
# lacks clear category cues, and deliberately produces the kind of honest
# errors required for the error-analysis section (rather than fabricating
# wrong predictions after the fact).
AMBIGUOUS_LABEL_TEMPLATES = [
    ("There is a problem in {area} that needs attention.",
     ["General", "Roads / Infrastructure", "Electricity", "Water Supply"]),
    ("Please resolve the issue in {area} as soon as possible.",
     ["General", "Sanitation", "Traffic", "Health"]),
    ("The situation in {area} has not improved despite complaints.",
     ["General", "Health", "Education", "Water Supply"]),
    ("No action has been taken in {area} for a long time.",
     ["General", "Roads / Infrastructure", "Water Supply", "Sanitation"]),
    ("This is a serious issue in {area}, please look into it urgently.",
     ["General", "Electricity", "Roads / Infrastructure", "Health"]),
    ("We have been facing this problem in {area} {dur}.",
     ["General", "Water Supply", "Electricity", "Sanitation"]),
]


def build_rows(target_total=850):
    rows = []
    # cycle through templates with area/duration substitution until target reached
    base_pool = []
    for category, sentiment, priority, templates in TEMPLATES:
        for t in templates:
            base_pool.append((category, sentiment, priority, t))

    n_ambiguous = 70
    idx = 0
    while len(rows) < target_total - len(ROMAN_URDU_TEMPLATES) * 6 - n_ambiguous:
        category, sentiment, priority, template = base_pool[idx % len(base_pool)]
        area = random.choice(AREAS)
        dur = random.choice(DURATIONS)
        text = template.format(area=area, dur=dur)
        rows.append((text, category, sentiment, priority, "en"))
        idx += 1

    # add Roman Urdu / mixed-language rows (repeated with variation)
    for _ in range(6):
        for category, sentiment, priority, template in ROMAN_URDU_TEMPLATES:
            area = random.choice(AREAS)
            dur = random.choice(DURATIONS)
            text = template.format(area=area, dur=dur)
            rows.append((text, category, sentiment, priority, "roman_ur"))

    # add genuinely ambiguous rows: same wording, randomly chosen plausible
    # category label, so the dataset contains irreducible classification
    # difficulty (used later for honest error analysis)
    for _ in range(n_ambiguous):
        template, possible_categories = random.choice(AMBIGUOUS_LABEL_TEMPLATES)
        category = random.choice(possible_categories)
        area = random.choice(AREAS)
        dur = random.choice(DURATIONS)
        text = template.format(area=area, dur=dur)
        rows.append((text, category, "Negative", "Medium", "en"))

    random.shuffle(rows)

    # de-duplicate exact text matches, keep first occurrence
    seen = set()
    deduped = []
    for r in rows:
        if r[0] not in seen:
            seen.add(r[0])
            deduped.append(r)
    return deduped


def main():
    rows = build_rows(target_total=850)

    raw_dir = os.path.join(os.path.dirname(__file__), "raw")
    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, "citizen_feedback_raw.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "category", "sentiment", "priority", "language"])
        for r in rows:
            writer.writerow(r)

    print(f"Generated {len(rows)} samples -> {out_path}")

    # quick class distribution summary
    from collections import Counter
    cat_counts = Counter(r[1] for r in rows)
    print("Category distribution:", dict(cat_counts))


if __name__ == "__main__":
    main()

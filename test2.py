from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

# faithfulness_metric = FaithfulnessMetric()
test_case = LLMTestCase(
    input=        """
I need to purchase in the next few hours, but I’ll be in meetings and can’t research. About me- I’ve ran 5 marathons, currently training for a spring marathon, should BQ, and I’ve only ran in NB, Saucony and Brooks. Volume is 30-40 mpw, 1-2 speed/tempo sessions, easy runs and LR. Female, 38 and thin. Any recs? I’ve never tried super shoes, but I’m totally open to trying them! I just don’t know which ones.

""" ,   actual_output="""
Hey there! It sounds like you're seriously dedicated to your marathon training, and it's awesome that you're considering trying out a pair of super shoes. Given your impressive running background and the volume of training you do, the Nike ZoomX Invincible Run 3 in White Cobalt Bliss could be a fantastic addition to your rotation.

Here's why these shoes might be just what you need: The ZoomX foam in the midsole is designed to give you incredible energy return, which is perfect for those speed and tempo sessions you're doing. It's like getting a little extra push with every stride, helping you maintain that pace you need to hit your Boston Qualifier (BQ) goal.

The Flyknit upper is not only super breathable but also strategically ventilated to keep your feet cool during those long runs. It'll ensure that you stay comfortable even as your mileage increases. Plus, the updated design has a sleek look, so you'll feel great wearing them both in training and on race day.

Stability is also key, especially with your mileage, and the redesigned heel clip offers just the right amount of support without adding bulk. This ensures you feel secure while still enjoying that cushioned, lively ride.

The outsole's lattice pattern is there to work with the foam, expanding and reacting to your movements, which means you'll have a responsive feel underfoot, whether you're on a recovery run or pushing the pace.""",
    expected_output="""Nike Structure
Vaporfly 
invincible
""",
retrieval_context=[
    """
The Nike Invincible Run 3 White Cobalt Bliss updates the Invincible Run silhouette adding even more stability and raising the bar on comfort and cushioning for every stride, all packaged in a sleeker design.\r\n\r\nThe upper of the White Cobalt Bliss Nike Invincible Run 3 sports a white Flyknit upper with breathability zones placed where your foot heats up the most to provide maximum comfort. A smaller heel clip than on previous Invincible iterations provides structure and support in a more streamlined design. An ice blue mud guard sports a tonal embossed Nike swoosh that matches the one on the heel. A black heel tab compliments a black striped Nike swoosh on the quarter panel and a solid black Nike swoosh on the toe box. This upper gets springy support from a wider midsole composed of cream-colored Zoom X foam that returns energy with every step.\r\n\r\nWhat we appreciate most about this high-octane running shoe is the black rubber outsole interwoven with hits of blue. The lattice pattern allows the foam to expand and react to empower your stride. The Nike Invincible Run 3 White Cobalt Bliss was released on January 5, 2023 for $180 retail"""
]
#     retrieval_context=[
#     {
#         "name": "Nike ZoomX Invincible Run 3 White Cobalt Bliss",
#         "description": "The Nike Invincible Run 3 White Cobalt Bliss updates the Invincible Run silhouette, adding even more stability and raising the bar on comfort and cushioning for every stride, all packaged in a sleeker design.\n\nThe upper of the White Cobalt Bliss Nike Invincible Run 3 sports a white Flyknit upper with breathability zones placed where your foot heats up the most to provide maximum comfort. A smaller heel clip than on previous Invincible iterations provides structure and support in a more streamlined design. An ice blue mud guard sports a tonal embossed Nike swoosh that matches the one on the heel. A black heel tab compliments a black striped Nike swoosh on the quarter panel and a solid black Nike swoosh on the toe box. This upper gets springy support from a wider midsole composed of cream-colored Zoom X foam that returns energy with every step.\n\nWhat we appreciate most about this high-octane running shoe is the black rubber outsole interwoven with hits of blue. The lattice pattern allows the foam to expand and react to empower your stride. The Nike Invincible Run 3 White Cobalt Bliss was released on January 5, 2023 for $180 retail.",
#         "color": "White",
#         "category": "Running",
#         "similarity_score": 0.5254333341680794,
#         "colorway": "White/Black-Football Grey-Cobalt Bliss-Pink Spell",
#         "release_date": "January 5, 2023",
#         "price": 180
#     }
# ]
)

metric = ContextualPrecisionMetric(threshold=0.5)

metric.measure(test_case)

# Print results
print("Precision Score:", metric.score)
print("Reason:", metric.reason)
print("Is Successful:", metric.is_successful())
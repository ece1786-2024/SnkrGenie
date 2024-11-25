from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

# faithfulness_metric = FaithfulnessMetric()
test_case = LLMTestCase(
    input="My current running shoes are pretty beat up and I'm looking for a new pair, but I have a problem with some running shoes. I usually wear shoes with good arch (normal, and I find Nike perfect), and stable midsole. They should have balance and midsole durability. By durability I mean those that not lean to one side over time. I have this problem with my right foot. (only model this did not happen was Pegasus 39). So anyone have any advice/suggestion on what I should pic (recommand a few because some maynot be available here). ",
    actual_output="""
Hey there! It sounds like you're on the hunt for a running shoe that offers excellent arch support, stability, and durability—especially on that right foot. Given your needs, the Nike ZoomX Invincible Run 3 in White Cobalt Bliss could be a perfect fit for you. I would like cushioning and lightweight features.
This shoe is all about providing stability and comfort that lasts. Its design includes a wider midsole made from ZoomX foam, which is incredibly responsive and helps return energy with every step. This feature not only enhances your running efficiency but also contributes to maintaining a balanced stride, addressing your concern about shoes leaning to one side over time.
The smaller heel clip in this model adds subtle structure and support without compromising its streamlined look, helping to keep your foot stable. Plus, the rubber outsole with its lattice pattern is designed to allow the foam to expand and react dynamically, giving you a reliable grip and further promoting durability.

The breathable Flyknit upper ensures your foot stays comfortable, even during tough runs, which is especially nice if you're logging those extra miles. With the Invincible Run 3, you'll find that the combination of cushioning, support, and durability is crafted to keep you running comfortably without the worry of uneven wear.
Hope this helps, and happy running with your new kicks!
""",
    expected_output="""Nike Structure
Air Max 270
Invincible Run 3 """,
retrieval_context=[
    """
232,Nike ZoomX Invincible Run 3 Black White,"The Nike ZoomX Invincible Run 3 Black White is part of the Nike ZoomX Invincible Run 3 lineup of running shoes. Although the colorway is Black, White, Dark Grey, and Coconut Milk, the Invincible Run 3 Black White presents as a dual-tone black and white sneaker.
This sneaker stands out from others due to the amount of cushioning underfoot. ZoomX foam cushioning in the midsole conforms to the foot shape and added foam height provides a soft underfoot feel. The lightweight Flyknit upper is both breathable and durable. In terms of branding, a Swoosh made of stripes graces the side panel, another Swoosh covers the vamp, and lettering along the heel spells out ‘NIKE INVINCIBLE RUN 3’. In addition, the midsole features an orange rectangle with Nike ZoomX foam branding.
What do our StockX experts love about this sneaker? It is comparatively light weighing approximately 310 grams. The Nike ZoomX Invincible Run 3 Black White was released on January 5, 2023, and retailed at $180.",Black/White-Dark Grey-White-Coconut Milk,Black,Running,180,Nike
"""
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
# metric2 = FaithfulnessMetric()
metric.measure(test_case)

# Print results
print("Precision Score:", metric.score)
print("Reason:", metric.reason)
print("Is Successful:", metric.is_successful())


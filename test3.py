from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

# faithfulness_metric = FaithfulnessMetric()
test_case = LLMTestCase(
    input=        """
I’m on my feet 🦶 for about 14 miles a day for 6 days a week. Does anyone know or any Nike enthusiast who can recommend me to some good sneakers? I currently only wear Pegasus or Pegasus trails and they seem to work wonders especially with my plantar fasciitis but I’m noticing that I’m breaking down the shoes a lot more faster than I thought. I like white and black style
""" ,   
    actual_output="""
Hey there! It sounds like you’ve got quite the mileage on your feet, and it’s awesome that the Pegasus line has been working wonders for you. Given your current needs, the Nike React Pegasus Trail 4 Gore-Tex Barely Grape Total Orange could be a fantastic match for your routine.
First off, since you're on your feet so much, the React midsole in this sneaker is a stellar feature. It offers exceptional cushioning and energy return, which means you'll feel that spring in your step even as you clock those 14 miles a day. This can be especially beneficial for managing plantar fasciitis, as it helps reduce the impact on your feet, making each step more comfortable and less stressful on your arches.
The Gore-Tex upper is another highlight, providing excellent water resistance. This is perfect if you're transitioning from indoors to outdoors or if your routes occasionally get a bit wet. No more worrying about soggy feet affecting your comfort or performance!
For durability, which seems to be a concern for you, the rugged design of the Trail 4 is built to withstand more intense wear. The black rubber outsole ensures reliable traction on a variety of surfaces, so whether you're on a trail or pavement, you can trust that your shoes will grip well and wear longer.
And style-wise, while the Barely Grape and Total Orange accents add a pop of color, the overall look still maintains a clean aesthetic with its white and black elements, aligning nicely with your color preference.
""",
    expected_output="""Nike Structure
invincible

""",
retrieval_context=[
    """
Nike React Pegasus Trail 4 Gore-Tex Barely Grape Total Orange', 'description': 'Name: Nike React Pegasus Trail 4 Gore-Tex Barely Grape Total Orange\n        Description: The Nike React Pegasus Trail 4 Gore-Tex Barely Grape Total Orange offers a trail running shoe silhouette enhanced with rugged all-weather fabric.\r\n\r\nThe Barely Grape Total Orange iteration of the Nike React Pegasus Trail 4 x Gore-Tex sports a pale purple mesh upper created with Gore-Tex to provide water resistance. Hits of green adorn the toe and textile collar that can be pulled tight with black laces. The heel and tongue each features a black pull tab to facilitate on/off. A yellow React midsole provides cushioning and energy return, while the black rubber outsole delivers traction that works on rugged terrain as well as on smooth pavement. The outer-side Nike Swoosh is outlined in orange, while a miniature green Nike Swoosh appears vertically on the collar. Another miniature Nike Swoosh in white appears on the inner side, while a yellow Nike Swoosh is debossed on the outsole.\r\n\r\nThe detail we love most about this sneaker is the motto on the midsole: JUST DO IT - REASON NOT REQUIRED. The Nike React Pegasus Trail 4 Gore-Tex Barely Grape Total Orange was released on January 5, 2023, for $160 retail
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

metric.measure(test_case)

# Print results
print("Precision Score:", metric.score)
print("Reason:", metric.reason)
print("Is Successful:", metric.is_successful())
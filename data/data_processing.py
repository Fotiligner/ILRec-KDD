import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import torch
import tqdm
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoConfig, AutoTokenizer, AutoModelForCausalLM

input_inter_file = "./LC-Rec/data/Instruments/Instruments.inter.json"
input_item_file = "./LC-Rec/data/Instruments/Instruments.item.json"

inter_dict = {}
item_dict = {}

with open(input_inter_file, "r") as file:
    inter_dict = json.load(file)

with open(input_item_file, "r") as file:
    item_dict = json.load(file)

all_inter_dict = {}

count = 0
for k, v in inter_dict.items():
    all_inter_list = []
    if len(v) >= 3:
        for index in range(1, len(v)):
            current_dict = {}
            current_dict["inter"] = v[:index]
            current_dict["target"] = v[index]

            all_inter_list.append(current_dict)
    
    all_inter_dict[k] = all_inter_list
    count += 1

train_inter = []
val_inter = []
test_inter = []


for k, v in all_inter_dict.items():
    train_list = v[:-2]
    val_list = v[-2]
    test_list = v[-1]

    for element in train_list:
        train_inter.append(element)

    val_inter.append(val_list)
    test_inter.append(test_list)

category = "instrument"

instructs = [
        f"Given a list of {category} the user recently enjoy, please recommend a new {category} that the user may bought",
        f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
        f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
        f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
        f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
        f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
        f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
        f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
        f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
        f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
        f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]


def generate_prompt(input, output):
        return f"""### User Input: 
{input}

### Response: 
{output}"""

def pre(row): # 这里row是包含inter和target两部分的
    instruction =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instructs[0]}
"""

    history = ""
    for i, item in enumerate(row["inter"]):
        if i == 0:
            history += "\"" + item_dict[str(item)]["title"] + "\""
        else:
            history += ", \"" + item_dict[str(item)]["title"] + "\""  
        
    target_item = "\"" + item_dict[str(row["target"])]["title"] + "\""

    input = f"The user has purchased the following {category}s before: {history}"
    output = ""
    prompt = generate_prompt(input, output)

    temp_str = ""
    temp_str = instruction + prompt

    # temp_str = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. \n\n### Instruction:\nGiven a list of art the user recently enjoy, please recommend a new art that the user may bought\n### User Input: \nThe user has watched the following arts before: \"Faber-Castel Extra Superfine PITT Artist Pen, Black.\", \"Winsor & Newton Water Color Blending Medium, 75ml.\", \"Winsor & Newton Series 7 Kolinsky Sable Watercolor Brush - Round #2.\", \"Winsor & Newton Series 7 Kolinsky Sable Watercolor Brush - Round #6.\", \"Bee Paper Company 808S60-909 Bee Paper Super Deluxe Sketch Pad, 9-Inch by 9-Inch.\", \"Winsor & Newton Series 7 Kolinsky Sable Watercolor Brush - Round #5.\", \"Assorted Blending Stumps and Tortillions, 6-count,6-piece Pencil and Pastel Blending Set.\", \"Pentel Arts Aquash Fine Point Water Brush, Medium (FRHMBP).\", \"Derwent Battery Operated Eraser, Artist Tool, Drawing, Art Supplies (2301931).\", \"Akashiya CA200/20V Sai Watercolor Brush Pen - 20 Color Set (1, DESIGN 1).\", \"Sugru SMLT3 3 Pieces Air-curing Rubber Set, 5g Capacity, Black/White/Red.\", \"JUST STOW IT 66-JS1702Just Stow It Easel Back Brush Case, Black.\", \"Penny Black PB30095 Flower Festival Stamps Sheet, 5 by 7.5-Inch, Clear.\", \"Sakura 38081 Pigma Blister Card Brush Pen, Black.\", \"Sakura 38061 6-Piece Pigma Assorted Colors Brush Pen Set />.\", \"M. Graham Tube Watercolor Paint Cityscape 5-Color Set, 1/2-Ounce.\", \"Ranger CND14737 Inkssentials Cut-N-Dry Nibs.\", \"ROYAL BRUSH 9-Inch by 13-Inch Graphite Paper, 20-Sheet, Grey.\", \"M. Graham Tube Watercolor Paint Landscape 5-Color Set, 1/2-Ounce.\", \"Stampers Anonymous Tim Holtz Cling Rubber Flower Garden Stamp Set, 7 x 8.5.\", \"M. Graham Tube Watercolor Paint Marinescape 5-Color Set, 1/2-Ounce.\", \"Gaunt Industries HYPO-200 - Oiler Boiler- Watercolor & Thin Acrylic Paint Applicator- 1-1/4 Ounce Clear Oval Plastic Bottle with 25 Gauge Blunt Tip- Precision Paint Dispenser- Use with Thin Liquids.\", \"QoR Watercolor Introductory 6-High Chroma Set.\", \"Ranger Emboss it Dabber Bottle, 1-Ounce, Clear.\", \"Scor-Pal SP108 Eighths Measuring and Scoring Board, 12 by 12, 1/8 Space Grooves.\", \"Inkadinkado Large Clear Block.\", \"Princeton Artist Brush Neptune, Brushes for Watercolor Series 4750, Round Synthetic Squirrel, Size 16.\", \"Soft Knife and Covers, 4 Assorted Knives and 8 Covers.\", \"Grafix Vellum, Assortment.\", \"Gaunt Industries HYPO-65 - Ceramic & Clay Underglaze Applicator - 2 Ounce Clear plastic Bottle with 16 Gauge Blunt Needle tip - Slip Trailing bottle.\", \"Califone 2924AVP-BL Deluxe Monaural Headset, Blueberry.\", \"Kuretake Zig Wink of Stella Glitter Brush Pen, Blue.\", \"Brusho by Colourcraft 8 Brusho Crystal Colour Set.\", \"Zig Memory System Wink of Stella Brush Glitter Markers, White Christmas, White, Red, Dark Green, 3-Pack.\", \"Locking Magnetic Clasps - Set of 4 GOLD By Jumbl.\", \"Panpastel Ultra Soft Artist Pastel Drawing Set (PPSTL10-30102).\", \"Gaunt Industries HYPO-49 - Craft Glue Applicator - 2 Ounce Clear Plastic Bottle with 18 Gauge Blunt Needle tip - Quilter's Basting glue dispenser.\", \"PanPastel 9-Milliliter Ultra Soft Artist Pastel Set, Blues, 5-Pack.\", \"Inkadinkado Clear Extra Large Block.\", \"Ampersand Aquabord 12 in. x 16 in. each.\", \"Derwent Blender and Burnisher Pencil Set, Drawing, Art Supplies (2301774).\", \"Panpastel Ultra Soft Artist Pastel Greens Set.\", \"Inkadinkado Clear Small And Medium Blocks.\", \"Scor-Pal SP202 Scor-Tape, 0.25 by 27-Yard.\", \"Handbook Paper Fluid 100 Watercolor Cp 300Lb Pochette White 8X10.\", \"Colorfin Sofft Applicators Replacement Heads.\", \"PanPastel 10 Cavity Palette Tray.\", \"HELMAR 450 Quick Dry Adhesive, 4.23 Fluid Ounce.\", \"PanPastel 9-Milliliter Ultra Soft Artist Pastel Set, Tint, 5-Pack.\", \"ZIG Wink of Stella Glitter Brush Marker Pen 999 Clear.\", \"Legion Yupo Polypropylene Pad, 5 X 7 inches, Medium 74lb, 10 Sheets (L21-YUP197WH57).\", \"Scotch Vellum Tape (005).\", \"Bill Buchman Zen Reed Pen 1 - Bamboo Pen - Drawing Pen - Fine.\", \"Ranger Embossing Powder, 1-Ounce Jar, Brown.\", \"Loew-Cornell 1023599 Soft Comfort Round Brush Set.\", \"PanPastel Ultra Soft Artist Pastel, Phthalo Green.\", \"Copic Markers 8-1/2 by 11-Inch Blending Card by X-Press It, 25 Sheets.\", \"Wow Embossing Powder 15ml-Clear Gloss.\", \"Wow Embossing Powder 15ml-Clear Gloss.\", \"Colorfin PanPastel Colorless Blender, 9ml.\", \"GANE ADH0901 Yes All-Purpose Stik Flat Glue, 1-Pint.\", \"Envelope Punch Board by We R Memory Keepers. The Easiest Envelope Maker Available.\", \"PanPastel Ultra Soft Artist Pastel, Ultramarine Blue Tint.\", \"Panpastel Ultra Soft Artist Pastel Shades Set.\", \"Santa Fe Art Supply Best Quality Artist Paintbrush Set. Acrylic Oil Watercolor & Face Paint. 15 (+1) Professional Paint Brushes In Travel Case.\", \"Jack Richeson Tom Lynch Porcelain Watercolor Palette.\", \"Colorfin Sofft Applicator and 4 Replacement Heads.\", \"Princeton Artist Brush Select Synthetic Brush Flat Wash 1 Width.\", \"Arches Aquarelle Watercolor Block 140 lb. cold press 7 in. x 10 in.\", \"Creative Mark Pastel Storage Box, Wooden 3 Drawer, Sturdy & Stackable, Perfect For Pastels, Art Tools, Paint Brushes & Makeup Brushes -Natural Finish 9\u00bd D \u00d7 16 W \u00d7 3\u00bc H.\", \"St Petersburg White Nights Watercolour : 36 Pan Set.\", \"Arches Watercolor Block, Cold Press, 9 x 12.\", \"Faber-Castel FC128272 Creative Studio Soft Pastel Crayons (72 Pack), Assorted.\", \"Jack Richeson Porcelain Palette 12 Well Round with Plastic Cover.\", \"Colorfin 30061 PanPastel Ultra Soft Artist Metallic Pastel Set, 9ml, Set of 6, 6-Pack.\", \"da Vinci Watercolor Series 5580 CosmoTop Spin Paint Brush, Round Synthetic with Red Handle, Size 12 (5580-12).\", \"Terry Harrison's Special Effects Brushes - Badger Blends (Medium).\", \"ZEM BRUSH Student Golden Synthetic Long Filbert - Cats Tongue Brush Set Sizes 4, 6, 8, 10.\", \"DANIEL SMITH 001900482 Watercolor 238 Dot Color Chart.\", \"Schmincke Horadam Aquarell Half-Pan Paint Metal Set with 12 Open Spaces, Set of 12 Colors (74412097).\", \"Kuretake Picture Letter Gansai Tanbi, 36 Color Set (MC20/36V).\", \"Ad-Tech 14ZIP50 Multi Temp Glue Stick (4 x 0.44-Inch), Pack of 50.\", \"Yasutomo Fold Ems Non-Toxic Square Origami Paper, 5-7/8 X 5-7/8 in, Assorted Metallic Color, Pack of 36.\", \"Aketek 6-Color Set Extra-LARGE 4 Steel Safety Pins - Blankets, Skirts, Kilts, Crafts.\", \"KLOUD City \u00ae6 Pcs 4 Inch Heavy Duty Giant Safety Pins for Blankets, Skirts, Kilts, Knitted Fabric ,Crafts (2silver&2black & 2bronze).\", \"Chalk Finish Paint Mix by Dover's \u2013 Add to Any Color of Flat Latex or Acrylic Paint to Make 2 Gallons of Inexpensive Chalk Furniture Paint - Save 75% Over All Natural, Non-Toxic.\", \"ROYAL BRUSH Watercolor Sponges 6/Pkg.\", \"Prima Marketing Tropical Watercolor Confections.\", \"Liquitex Gloss Acrylic Fluid Medium and Varnish, 8-oz.\", \"Liquitex Professional Black Gesso Surface Prep Medium Bottle, 8-Ounce.\", \"Martha Stewart Crafts Fine Glitter, Crystal, 1-1/2 Ounces />.\", \"Crystal Diamante Corsage Pins pk/100.\"\n\n### Response: \n"

    # model_name = "/home/hadoop-ba-dealrank/dolphinfs_hdd_hadoop-ba-dealrank/libingqian/llama3/Llama-3-8B/LLM-Research/Meta-Llama-3-8B"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token_id = (0)
    # tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    # outputing = tokenizer(temp_str, return_tensors='pt')


    return dict(
        input=instruction + prompt,
        chosen=target_item
    )


final_output = []
for inter in tqdm.tqdm(train_inter):
    final_output.append(pre(inter))

    # if length > maxing:
    #     print(f"{length} {i}")

with open("./Instruments/train_instruments_bigrec_data.json", "w") as file:
    json.dump(final_output, file, indent=4)

# val
final_output = []
for inter in tqdm.tqdm(val_inter):
    final_output.append(pre(inter))

with open("./Instruments/val_instruments_bigrec_data.json", "w") as file:
    json.dump(final_output, file, indent=4)

# test
final_output = []
for inter in tqdm.tqdm(test_inter):
    final_output.append(pre(inter))

with open("./Instruments/test_instruments_bigrec_data.json", "w") as file:
    json.dump(final_output, file, indent=4)






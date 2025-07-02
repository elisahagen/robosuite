# instruction_templates.py

# Level 1: General task description (from A to B)
InstructionTemplatesLevel1 = {
    "bread": [
        "Pick up the bread and place it in the bin.",
        "Move the bread from one location to another.",
        "Transfer the bread to the target area.",
        "Grab the bread and drop it somewhere in the bin."
    ],
    "milk": [
        "Pick up the milk and place it in the bin.",
        "Move the milk to the other side.",
        "Grab the milk and drop it in the bin area."
    ],
    "box": [
        "Move the box to the other side.",
        "Place the box in the target bin.",
        "Pick up the box and transfer it."
    ],
    "cereals": [
        "Grab the cereal box and move it to the bin.",
        "Transfer the cereal box to the destination.",
        "Pick up and place the cereals in the bin."
    ],
}

# Level 2: Use precise spatial coordinates (based on env.target_position)
InstructionTemplatesLevel2 = {
    obj: [
        f"Pick up the {obj} and place it at coordinates {{}}.",
        f"Move the {obj} to the position located at {{}}.",
        f"Transfer the {obj} to the exact spot at {{}}."
    ] for obj in InstructionTemplatesLevel1
}

# Level 3: Natural language spatial references (free space/corners)
InstructionTemplatesLevel3 = {
    "bread": [
        "Pick up the bread and place it into the empty space next to the cubes.",
        "Move the bread into the unoccupied corner of the bin.",
        "Place the bread into the last available slot between the other cubes.",
        "Grab the bread and set it into the vacant area of the bin."
    ],
    "milk": [
        "Pick up the milk and place it into the empty space next to the cubes.",
        "Move the milk into the unoccupied corner of the bin.",
        "Place the milk into the last available slot between the other cubes.",
        "Grab the milk and set it into the vacant area of the bin."
    ],
    "box": [
        "Pick up the box and place it into the empty space next to the cubes.",
        "Move the box into the unoccupied corner of the bin.",
        "Place the box into the last available slot between the other cubes.",
        "Grab the box and set it into the vacant area of the bin."
    ],
    "cereals": [
        "Pick up the cereals and place it into the empty space next to the cubes.",
        "Move the cereals into the unoccupied corner of the bin.",
        "Place the cereals into the last available slot between the other cubes.",
        "Grab the cereals and set it into the vacant area of the bin."
    ],
}

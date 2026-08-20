import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/macbooka/Downloads/maniskill_instructions_to_label.xlsx";
const outputDir = "/Users/macbooka/Documents/VsCode/multimodal/outputs/task_router_300_20260819";
const workDir = "/Users/macbooka/Documents/VsCode/multimodal/.codex_work/task_router_300";

const groups = [
  {
    id: 1,
    mode: "navigation",
    episodes: 2467,
    steps: 345243,
    note: "Move an unspecified object to a goal position; label inherited from source task.",
    instructions: [
      "pick up the object and move it to a goal position.",
      "pick up the object and place it at the goal position.",
      "grasp the object and move it to the target position.",
      "lift the object and carry it to the goal location.",
      "move the object to its designated goal position.",
      "take the object and put it at the target location.",
      "grab the object and relocate it to the goal.",
      "place the object in the specified goal position.",
      "pick the object up and set it down at the target.",
      "transport the object to the indicated goal location.",
      "grasp and transfer the object to its target position.",
      "lift the object, then place it at the goal.",
      "move the selected object into the goal position.",
      "pick up the item and bring it to the target spot.",
      "relocate the object from its current position to the goal.",
      "carry the object over to the designated target position.",
      "take hold of the object and place it at the goal location.",
      "pick and place the object at the indicated target.",
      "move the item until it is positioned at the goal.",
      "grasp the item and set it at the specified destination.",
      "lift and reposition the object at the target location.",
      "transfer the object to the marked goal position.",
      "pick the object up and move it into the target area.",
      "bring the object to its assigned goal position.",
      "move the object from where it is to the target spot.",
      "grasp the object and place it in the goal location.",
      "pick up the item and relocate it to its target.",
      "carry the object to the specified destination.",
      "lift the object and position it at the indicated goal.",
      "take the object to the goal and place it there."
    ],
  },
  {
    id: 2,
    mode: "object_state",
    episodes: 806,
    steps: 128662,
    note: "Identify and pick the designated object from clutter; label inherited from source task.",
    instructions: [
      "pick up a designated object from a clutter of objects.",
      "pick up the specified object from among the clutter.",
      "grasp the designated item in the group of objects.",
      "select the indicated object from the clutter and pick it up.",
      "find the designated object among the other items and lift it.",
      "identify and grasp the specified object in the clutter.",
      "pick the requested object out of the collection.",
      "locate the designated item among the objects and pick it up.",
      "choose the specified object from the clutter and grasp it.",
      "lift the indicated object from the crowded set of items.",
      "retrieve the designated object from among the surrounding objects.",
      "grasp only the specified item from the cluttered group.",
      "pick out and lift the designated object.",
      "from the clutter, select the indicated object and pick it up.",
      "find the requested item in the group and grasp it.",
      "identify the target object among the clutter and lift it.",
      "take hold of the designated object from the collection.",
      "pick up the particular object indicated within the clutter.",
      "locate and grasp the specified item among the other objects.",
      "select the target item from the cluttered objects and lift it.",
      "retrieve the indicated object without picking the surrounding items.",
      "pick the designated item from the assortment of objects.",
      "search the clutter for the specified object and grasp it.",
      "lift the requested object from the group of items.",
      "choose and pick up the designated object among the clutter.",
      "find the indicated object in the assortment and lift it.",
      "grasp the requested item from the cluttered collection.",
      "identify the designated object and pick it out of the group.",
      "pick up the target object from among the other objects.",
      "locate the specified item in the clutter and take hold of it."
    ],
  },
  {
    id: 3,
    mode: "default",
    episodes: 773,
    steps: 141707,
    note: "Turn on a faucet by rotating its designated handle; label inherited from source task.",
    instructions: [
      "turn on the faucet by rotating a designated handle.",
      "rotate the designated handle to turn on the faucet.",
      "turn the specified faucet handle until the faucet is on.",
      "activate the faucet by rotating the indicated handle.",
      "use the designated handle to switch the faucet on.",
      "grasp and rotate the specified handle to turn on the faucet.",
      "turn on the water by rotating the designated faucet handle.",
      "rotate the indicated handle so that the faucet turns on.",
      "operate the designated handle to activate the faucet.",
      "switch on the faucet using the specified rotating handle.",
      "twist the designated faucet handle to the on position.",
      "turn the indicated handle to activate the water flow.",
      "rotate the selected handle and turn the faucet on.",
      "use the indicated faucet handle to start the water.",
      "activate the water flow by turning the designated handle.",
      "grasp the specified handle and rotate it to switch on the faucet.",
      "move the designated faucet handle by rotation until it is on.",
      "turn the chosen handle to start the faucet.",
      "rotate the requested handle to open the faucet.",
      "switch the faucet on by turning the indicated handle.",
      "twist the specified handle until water is turned on.",
      "operate the faucet by rotating its designated handle to on.",
      "turn the designated control handle to activate the faucet.",
      "rotate the correct faucet handle to start the water flow.",
      "use the specified handle and turn it until the faucet activates.",
      "start the faucet by rotating the designated handle.",
      "turn the indicated faucet control into the on position.",
      "rotate the designated control to switch on the water.",
      "activate the faucet using the indicated rotating handle.",
      "grip and turn the specified handle so the faucet comes on."
    ],
  },
  {
    id: 4,
    mode: "navigation",
    episodes: 276,
    steps: 42777,
    note: "Insert a designated object into its corresponding board slot; label inherited from source task.",
    instructions: [
      "insert a designated object into the corresponding slot on a board.",
      "place the designated object into its matching slot on the board.",
      "insert the specified item into the corresponding board slot.",
      "fit the designated object into the slot that matches it on the board.",
      "put the indicated object into its corresponding slot in the board.",
      "align the specified object with its matching slot and insert it.",
      "place the selected object inside the appropriate slot on the board.",
      "insert the indicated item into the board opening that corresponds to it.",
      "fit the specified object into its designated board slot.",
      "match the object to its slot on the board and insert it.",
      "put the designated piece into the matching opening on the board.",
      "guide the specified object into its corresponding board slot.",
      "insert the selected piece into the appropriate slot in the board.",
      "place the indicated item into the board slot assigned to it.",
      "fit the designated piece inside its matching board opening.",
      "align and insert the requested object into the corresponding slot.",
      "move the specified object into its matching slot on the board.",
      "put the selected item in the correct corresponding board slot.",
      "insert the designated shape into the slot that matches it.",
      "place the requested object into its proper opening on the board.",
      "guide the indicated piece into the corresponding board opening.",
      "fit the selected object into the correct slot on the board.",
      "insert the requested item into its matching board position.",
      "place the designated piece securely in its corresponding slot.",
      "align the indicated object to the right board slot and insert it.",
      "move the selected piece into the board opening made for it.",
      "put the specified shape into its corresponding slot on the board.",
      "insert the indicated object into the correct matching opening.",
      "fit the requested piece into its assigned slot in the board.",
      "place the specified object within the corresponding board slot."
    ],
  },
  {
    id: 5,
    mode: "navigation",
    episodes: 186,
    steps: 30023,
    note: "Plug a charger into a wall socket; label inherited from source task.",
    instructions: [
      "plug the charger into the wall socket.",
      "insert the charger plug into the wall socket.",
      "connect the charger to the wall outlet.",
      "plug the charger into the electrical outlet on the wall.",
      "place the charger plug into the wall receptacle.",
      "insert the charger's plug into the socket.",
      "connect the charger by plugging it into the wall outlet.",
      "fit the charger plug into the wall socket.",
      "put the charger connector into the wall outlet.",
      "plug the charging adapter into the wall socket.",
      "align the charger plug with the wall outlet and insert it.",
      "insert the charging plug into the electrical socket.",
      "connect the charging cable to the wall socket using its plug.",
      "place the charger into the available wall outlet.",
      "push the charger plug into the wall receptacle.",
      "attach the charger to power by plugging it into the wall socket.",
      "insert the charger connector into the electrical outlet.",
      "plug the power adapter into the socket on the wall.",
      "connect the charger plug securely to the wall outlet.",
      "guide the charger plug into the wall socket.",
      "put the charging adapter's plug into the outlet.",
      "fit the charging plug into the electrical wall socket.",
      "insert the charger into the wall outlet to connect it.",
      "plug the charger connector into the socket.",
      "place the charger plug inside the wall outlet.",
      "connect the charger to the electrical socket.",
      "align and insert the charging adapter into the wall receptacle.",
      "push the charger connector into the wall outlet.",
      "plug the power charger into the electrical wall socket.",
      "insert the charging adapter's plug into the socket on the wall."
    ],
  },
  {
    id: 6,
    mode: "navigation",
    episodes: 178,
    steps: 26009,
    note: "Stack the red cube directly on the green cube; label inherited from source task.",
    instructions: [
      "stack the red cube on top of the green cube.",
      "place the red cube on top of the green cube.",
      "put the red cube above and directly onto the green cube.",
      "set the red cube on the upper surface of the green cube.",
      "lift the red cube and stack it on the green cube.",
      "position the red cube on top of the green one.",
      "place the red block directly over the green block to form a stack.",
      "stack the cubes with red on top of green.",
      "put the red block onto the top face of the green block.",
      "move the red cube onto the green cube and stack them.",
      "arrange the cubes so the red cube rests on the green cube.",
      "set the red block on top of the green block.",
      "place the green cube underneath the red cube by stacking the red one on it.",
      "make a stack with the red cube above the green cube.",
      "position the red cube securely on the green cube.",
      "lift and place the red block on the top of the green block.",
      "put the red cube directly on the green cube's top surface.",
      "stack red over green, with the red cube on top.",
      "move the red block onto the green block to create a two-cube stack.",
      "place the red cube atop the green cube.",
      "arrange the red cube so it rests above the green cube.",
      "set the red cube down on the top face of the green cube.",
      "form a stack by placing red on green.",
      "put the red cube on the green cube without reversing their order.",
      "stack the red block above the green block.",
      "position red as the upper cube and green as the lower cube.",
      "place the red cube over the green cube to make a stable stack.",
      "move the red cube onto the top of the green cube.",
      "build a two-cube stack with green below red.",
      "set the red cube on the green one so that red is on top."
    ],
  },
  {
    id: 7,
    mode: "navigation",
    episodes: 155,
    steps: 23988,
    note: "Insert the peg into the horizontal hole in a box; label inherited from source task.",
    instructions: [
      "insert the peg into the horizontal hole in a box.",
      "put the peg into the box's horizontal hole.",
      "insert the peg through the horizontally oriented hole in the box.",
      "align the peg with the horizontal opening and push it into the box.",
      "place the peg inside the horizontal hole on the box.",
      "fit the peg into the box opening that runs horizontally.",
      "guide the peg into the horizontal hole in the box.",
      "push the peg into the horizontally aligned box hole.",
      "insert the peg into the box through its horizontal opening.",
      "align and place the peg in the box's horizontal socket.",
      "fit the peg inside the horizontal opening of the box.",
      "put the peg through the hole oriented horizontally on the box.",
      "move the peg into alignment with the horizontal hole and insert it.",
      "guide the peg through the box's horizontally oriented opening.",
      "place the peg into the horizontal socket on the box.",
      "insert the peg straight into the horizontal box opening.",
      "push the peg through the horizontal hole in the box.",
      "fit the peg into the horizontally positioned hole.",
      "align the peg to the box's horizontal hole and place it inside.",
      "put the peg inside the opening that is horizontal on the box.",
      "insert the peg into the correct horizontal opening of the box.",
      "guide the peg into the hole on the box that faces horizontally.",
      "place the peg through the box's horizontal aperture.",
      "fit the peg securely into the horizontal hole in the box.",
      "move the peg into the box through the horizontally aligned hole.",
      "insert the peg in the box's side-facing horizontal opening.",
      "push the peg inside the horizontal socket of the box.",
      "align the peg horizontally with the box hole and insert it.",
      "put the peg through the horizontally oriented opening in the box.",
      "guide and insert the peg into the box's horizontal hole."
    ],
  },
  {
    id: 8,
    mode: "navigation",
    episodes: 143,
    steps: 14254,
    note: "Move the red cube to a goal position; label inherited from source task.",
    instructions: [
      "pick up the red cube and move it to a goal position.",
      "pick up the red cube and place it at the goal position.",
      "grasp the red cube and move it to the target location.",
      "lift the red cube and carry it to the goal.",
      "move the red cube into its designated target position.",
      "take the red cube and put it at the goal location.",
      "grab the red cube and relocate it to the target.",
      "place the red cube in the specified goal position.",
      "pick the red cube up and set it down at the target spot.",
      "transport the red cube to the indicated goal location.",
      "grasp and transfer the red cube to its target position.",
      "lift the red cube, then place it at the goal.",
      "move the selected red cube into the goal position.",
      "pick up the red block and bring it to the target spot.",
      "relocate the red cube from its current position to the goal.",
      "carry the red cube over to the designated target position.",
      "take hold of the red cube and place it at the goal location.",
      "pick and place the red cube at the indicated target.",
      "move the red block until it is positioned at the goal.",
      "grasp the red cube and set it at the specified destination.",
      "lift and reposition the red cube at the target location.",
      "transfer the red cube to the marked goal position.",
      "pick the red cube up and move it into the target area.",
      "bring the red cube to its assigned goal position.",
      "move the red block from where it is to the target spot.",
      "grasp the red cube and place it in the goal location.",
      "pick up the red cube and relocate it to its target.",
      "carry the red cube to the specified destination.",
      "lift the red block and position it at the indicated goal.",
      "take the red cube to the goal and place it there."
    ],
  },
  {
    id: 9,
    mode: "navigation",
    episodes: 16,
    steps: 1440,
    note: "Raise the red cube vertically by exactly 0.2 meters; label inherited from source task.",
    instructions: [
      "lift up the red cube by 0.2 meters.",
      "raise the red cube by 0.2 meters.",
      "lift the red cube 0.2 meters upward.",
      "move the red cube upward by 0.2 meters.",
      "raise the red block exactly 0.2 meters.",
      "pick up the red cube and elevate it by 0.2 meters.",
      "increase the red cube's height by 0.2 meters.",
      "lift the red block vertically through a distance of 0.2 meters.",
      "move the red cube 0.2 meters higher.",
      "elevate the red cube by twenty centimeters.",
      "raise the red cube vertically by 20 centimeters.",
      "lift the red block upward a distance of twenty centimeters.",
      "move the red cube straight up by 0.2 meters.",
      "raise the height of the red cube by 0.2 meters.",
      "pick the red cube up by twenty centimeters.",
      "elevate the red block exactly 0.2 meters above its current height.",
      "lift the red cube until it is 0.2 meters higher.",
      "move the red block vertically upward by twenty centimeters.",
      "raise the red cube from its current position by 0.2 meters.",
      "lift the red cube straight upward by twenty centimeters.",
      "elevate the cube colored red by 0.2 meters.",
      "move the red cube up exactly 20 centimeters.",
      "raise the red block 0.2 meters above where it is now.",
      "lift the red cube through a vertical displacement of 0.2 meters.",
      "increase the red block's vertical position by twenty centimeters.",
      "pick up and raise the red cube by 0.2 meters.",
      "elevate the red cube vertically by exactly twenty centimeters.",
      "move the red block upward until it is 0.2 meters higher.",
      "lift the red cube to a point twenty centimeters above its start.",
      "raise the red cube by a vertical distance of 0.2 meters."
    ],
  },
  {
    id: 10,
    mode: "temporal",
    episodes: null,
    steps: null,
    note: "Synthetic temporal-memory task requiring recall of the red cube's immediately previous position.",
    instructions: [
      "return the red cube to the position it occupied before it was moved.",
      "move the red cube back to its previous position.",
      "place the red cube at the location it occupied before its last move.",
      "restore the red cube to its prior location.",
      "take the red cube back to where it was previously.",
      "relocate the red cube to the position it held before being moved.",
      "put the red cube back in its earlier location.",
      "return the red block to the spot it occupied immediately before the current one.",
      "move the red cube from its current location back to its preceding position.",
      "restore the red block to its last position before relocation.",
      "place the cube colored red where it was before the most recent move.",
      "take the red block back to its previous recorded location.",
      "move the red cube back to the location it came from.",
      "return the red block to its earlier position before the transfer.",
      "reposition the red cube at the location it occupied one move ago.",
      "put the red cube back where it had been before.",
      "restore the red cube to the position preceding its current position.",
      "move the red block back to its last known position before relocation.",
      "return the red cube to the location from which it was moved.",
      "place the red cube in the prior position remembered from before the move.",
      "undo the most recent move by returning the red cube to its earlier location.",
      "reverse the red cube's last relocation and restore its prior position.",
      "move the red cube back to the spot it occupied previously.",
      "return the red block to the location it was in before its latest movement.",
      "restore the red cube to the exact position it had before it moved.",
      "place the red cube back at its former location.",
      "take the red cube to its immediately previous position.",
      "return the red block to the starting point of its most recent move.",
      "move the red cube back to where it was just before relocation.",
      "restore the red cube to the position it occupied prior to its current location."
    ],
  },
];

for (const group of groups) {
  if (group.instructions.length !== 30) throw new Error(`Group ${group.id} has ${group.instructions.length} instructions`);
}

const normalized = new Set();
for (const group of groups) {
  for (const instruction of group.instructions) {
    const key = instruction.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
    if (normalized.has(key)) throw new Error(`Duplicate instruction: ${instruction}`);
    normalized.add(key);
  }
}

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const source = workbook.worksheets.getItemAt(0);
source.name = "Source Tasks";
source.showGridLines = false;
source.freezePanes.freezeRows(1);
source.getRange("E2:E10").values = source.getRange("E2:E10").values.map(([value]) => [value === "spatial" ? "navigation" : value]);
source.getRange("A11:I11").values = [[10, groups[9].instructions[0], null, null, "temporal", "", "high", groups[9].note, "train"]];
source.getRange("A1:I11").format = { font: { name: "Aptos", size: 10 }, verticalAlignment: "center" };
source.getRange("A1:I1").format = { fill: "#17365D", font: { name: "Aptos", size: 10, bold: true, color: "#FFFFFF" }, rowHeight: 26 };
source.getRange("A2:I11").format.borders = { insideHorizontal: { style: "thin", color: "#D9E2F3" } };
source.getRange("A:A").format.columnWidth = 14;
source.getRange("B:B").format.columnWidth = 62;
source.getRange("C:D").format.columnWidth = 15;
source.getRange("E:F").format.columnWidth = 18;
source.getRange("G:G").format.columnWidth = 13;
source.getRange("H:H").format.columnWidth = 45;
source.getRange("I:I").format.columnWidth = 12;
source.getRange("B2:B11").format.wrapText = true;
source.getRange("H2:H11").format.wrapText = true;
source.getRange("E2:E11").dataValidation = { rule: { type: "list", values: ["navigation", "object_state", "default", "temporal"] } };

const readme = workbook.worksheets.add("README");
const augmented = workbook.worksheets.add("Augmented Instructions");
const summary = workbook.worksheets.add("Class Summary");

readme.showGridLines = false;
readme.getRange("A1:B1").values = [["Field", "Value"]];
readme.getRange("A2:B12").values = [
  ["Purpose", "Training-only paraphrase corpus for the instruction router"],
  ["Source tasks", 10],
  ["Phrasings per source task", 30],
  ["Total instructions", 300],
  ["Labels", "Inherited exactly from Source Tasks: navigation, object_state, default, temporal"],
  ["Split policy", "All rows are train. Do not validate on paraphrases of these same ten source tasks."],
  ["Imbalance mitigation", "Use sample_weight from Augmented Instructions or an equivalent weighted sampler."],
  ["Temporal class", "Supported by one synthetic source task with 30 reviewed paraphrases requiring recall of a previous object position."],
  ["Evaluation requirement", "Use unseen task families or another dataset; split by source task, never by paraphrase row."],
  ["Review status", "All paraphrases were checked for preservation of object, relation, direction, quantity, and action."],
  ["Important limitation", "300 rows represent only 10 independent task meanings; paraphrasing adds wording diversity, not task diversity."]
];
readme.getRange("A1:B1").format = { fill: "#17365D", font: { name: "Aptos", size: 11, bold: true, color: "#FFFFFF" }, rowHeight: 28 };
readme.getRange("A2:A12").format = { fill: "#D9EAF7", font: { name: "Aptos", size: 10, bold: true, color: "#17365D" } };
readme.getRange("B2:B12").format = { font: { name: "Aptos", size: 10 }, wrapText: true };
readme.getRange("A1:B12").format.borders = { insideHorizontal: { style: "thin", color: "#D9E2F3" }, outside: { style: "thin", color: "#9FBAD0" } };
readme.getRange("A:A").format.columnWidth = 25;
readme.getRange("B:B").format.columnWidth = 100;
readme.getRange("2:12").format.rowHeight = 34;

const headers = ["augmentation_id", "source_instruction_id", "group_id", "instruction", "is_original", "primary_mode", "secondary_mode", "confidence", "review_status", "split", "episode_count", "step_count", "sample_weight", "notes"];
const rows = [];
let augmentationId = 1;
for (const group of groups) {
  group.instructions.forEach((instruction, index) => {
    rows.push([
      augmentationId++,
      group.id,
      `maniskill_task_${group.id}`,
      instruction,
      index === 0,
      group.mode,
      "",
      index === 0 ? "source_label" : "inherited",
      index === 0 ? "source" : "assistant_reviewed",
      "train",
      group.episodes,
      group.steps,
      null,
      group.note,
    ]);
  });
}

augmented.getRange("A1:N271").values = [headers, ...rows];
augmented.getRange("M2").formulas = [["=IF(F2=\"spatial\",'Class Summary'!$D$2,IF(F2=\"object_state\",'Class Summary'!$D$3,IF(F2=\"default\",'Class Summary'!$D$4,\"\")))"]];
augmented.getRange("M2:M271").fillDown();
augmented.showGridLines = false;
augmented.freezePanes.freezeRows(1);
augmented.freezePanes.freezeColumns(3);
augmented.getRange("A1:N1").format = { fill: "#17365D", font: { name: "Aptos", size: 10, bold: true, color: "#FFFFFF" }, rowHeight: 28, wrapText: true };
augmented.getRange("A2:N271").format = { font: { name: "Aptos", size: 10 }, verticalAlignment: "center" };
augmented.getRange("A2:N271").format.borders = { insideHorizontal: { style: "thin", color: "#E7EEF5" } };
augmented.getRange("A:A").format.columnWidth = 16;
augmented.getRange("B:B").format.columnWidth = 20;
augmented.getRange("C:C").format.columnWidth = 20;
augmented.getRange("D:D").format.columnWidth = 78;
augmented.getRange("E:E").format.columnWidth = 12;
augmented.getRange("F:G").format.columnWidth = 18;
augmented.getRange("H:I").format.columnWidth = 20;
augmented.getRange("J:J").format.columnWidth = 10;
augmented.getRange("K:L").format.columnWidth = 15;
augmented.getRange("M:M").format.columnWidth = 15;
augmented.getRange("N:N").format.columnWidth = 68;
augmented.getRange("D2:D271").format.wrapText = true;
augmented.getRange("N2:N271").format.wrapText = true;
augmented.getRange("K2:L271").format.numberFormat = "#,##0";
augmented.getRange("M2:M271").format.numberFormat = "0.000";
augmented.getRange("F2:F271").dataValidation = { rule: { type: "list", values: ["spatial", "object_state", "default", "temporal"] } };
augmented.getRange("I2:I271").dataValidation = { rule: { type: "list", values: ["source", "assistant_reviewed", "needs_review"] } };
const augmentedTable = augmented.tables.add("A1:N271", true, "AugmentedInstructionsTable");
augmentedTable.style = "TableStyleMedium2";
augmentedTable.showFilterButton = true;

summary.showGridLines = false;
summary.getRange("A1:D1").values = [["class", "row_count", "share", "recommended_class_weight"]];
summary.getRange("A2:A5").values = [["spatial"], ["object_state"], ["default"], ["temporal"]];
summary.getRange("A6:A8").values = [["total"], [""], ["observed_classes"]];
summary.getRange("B2").formulas = [["=COUNTIF('Augmented Instructions'!$F$2:$F$271,A2)"]];
summary.getRange("B2:B5").fillDown();
summary.getRange("B6").formulas = [["=SUM(B2:B5)"]];
summary.getRange("B8").formulas = [["=COUNTIF(B2:B5,\">0\")"]];
summary.getRange("C2").formulas = [["=B2/$B$6"]];
summary.getRange("C2:C5").fillDown();
summary.getRange("D2").formulas = [["=IF(B2=0,\"N/A\",$B$6/($B$8*B2))"]];
summary.getRange("D2:D5").fillDown();
summary.getRange("A1:D1").format = { fill: "#17365D", font: { name: "Aptos", size: 10, bold: true, color: "#FFFFFF" }, rowHeight: 28, wrapText: true };
summary.getRange("A2:D8").format = { font: { name: "Aptos", size: 10 } };
summary.getRange("A6:D6").format = { fill: "#D9EAF7", font: { name: "Aptos", size: 10, bold: true, color: "#17365D" }, borders: { top: { style: "medium", color: "#9FBAD0" } } };
summary.getRange("A8:D8").format = { fill: "#FFF2CC", font: { name: "Aptos", size: 10, bold: true, color: "#7F6000" } };
summary.getRange("A:A").format.columnWidth = 22;
summary.getRange("B:B").format.columnWidth = 15;
summary.getRange("C:C").format.columnWidth = 13;
summary.getRange("D:D").format.columnWidth = 28;
summary.getRange("B2:B8").format.numberFormat = "#,##0";
summary.getRange("C2:C5").format.numberFormat = "0.0%";
summary.getRange("D2:D5").format.numberFormat = "0.000";
summary.freezePanes.freezeRows(1);

await fs.mkdir(outputDir, { recursive: true });

const checks = await workbook.inspect({
  kind: "table",
  range: "'Class Summary'!A1:D8",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 6,
  maxChars: 8000,
});
console.log(checks.ndjson);

const dataCheck = await workbook.inspect({
  kind: "table",
  range: "'Augmented Instructions'!A1:N12",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 14,
  maxChars: 12000,
});
console.log(dataCheck.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const previews = [
  ["Source Tasks", "A1:I10", "source_tasks_preview.png"],
  ["README", "A1:B12", "readme_preview.png"],
  ["Augmented Instructions", "A1:N22", "augmented_preview.png"],
  ["Class Summary", "A1:D8", "class_summary_preview.png"],
];
for (const [sheetName, range, fileName] of previews) {
  const preview = await workbook.render({ sheetName, range, scale: 1.25, format: "png" });
  await fs.writeFile(`${workDir}/${fileName}`, new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(`${outputDir}/maniskill_router_paraphrases_270.xlsx`);
console.log(`output=${outputDir}/maniskill_router_paraphrases_270.xlsx`);

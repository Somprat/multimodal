import csv

num_episodes = 96
rows = []


for i, episode in data:
    instruction = episode["instruction"]
    instruction = instruction.strip().lower()

    list_of_instructions = [memory.instructon for memory in data]

    if i in range(len(list_of_instructions)):
        if list_of_instructions[i] !=instruction:
            continue

    
        data.pop(i)

with open('output.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["instructions"])
    writer.writerow( wrtiter.right)
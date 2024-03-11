from simpful import *

# A simple fuzzy inference system for the tipping problem using TSK inference
# Create a fuzzy system object
FS = FuzzySystem()

# Define fuzzy sets for "Service"
S_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=5), term="poor")
S_2 = FuzzySet(function=Triangular_MF(a=0, b=5, c=10), term="good")
S_3 = FuzzySet(function=Triangular_MF(a=5, b=10, c=10), term="excellent")
FS.add_linguistic_variable("Service", LinguisticVariable([S_1, S_2, S_3], concept="Service quality", universe_of_discourse=[0,10]))

# Define fuzzy sets for "Food"
F_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="rancid")
F_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=10), term="delicious")
FS.add_linguistic_variable("Food", LinguisticVariable([F_1, F_2], concept="Food quality", universe_of_discourse=[0,10]))

# Define crisp output values for "Tip"
FS.set_crisp_output_value("small_tip", 0)
FS.set_crisp_output_value("average_tip", 10)
FS.set_crisp_output_value("generous_tip", 20)

# Define TSK fuzzy rules
R1 = "IF (Service IS poor) OR (Food IS rancid) THEN (Tip IS small_tip)"
R2 = "IF (Service IS good) THEN (Tip IS average_tip)"
R3 = "IF (Service IS excellent) OR (Food IS delicious) THEN (Tip IS generous_tip)"
FS.add_rules([R1, R2, R3])

# Set antecedents values
FS.set_variable("Service", 4)
FS.set_variable("Food", 8)

# Perform TSK inference and print output
result = FS.Sugeno_inference(["Tip"])
print(result)

print(FS.get_firing_strengths())

FS.produce_figure()
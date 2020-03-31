import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from textgenrnn import textgenrnn

print("Loading model...")
t = textgenrnn("textgenrnn_weights.hdf5")

# t = textgenrnn()
# print("Training model...")
# t.train_from_file("real.txt", num_epochs=10)


print("Loading existing names...")
with open("real.txt") as f:
    lines = f.read().split("\n")


def generate():
    g = t.generate(temperature=0.6, return_as_list=True)
    if g[0] in lines:
        #print(" --- Already exists ---")
        g = generate()

    return g[0]


generated = []
f_out = open("fake.txt", "a")

print("Generating\n")
print(str(len(lines)) + " lines to generate")

i = 0
while i < len(lines):
    name = generate()
    if len(name) == 1:
        #print(" --- Too short ---")
        continue

    if name in generated:
        #print(" --- Already generated ---")
        continue

    generated.append(name)
    f_out.write(name+"\n")
    print(name)
    i += 1

print("Saving...")

f_out.close()

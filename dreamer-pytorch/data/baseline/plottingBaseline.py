import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    with open("dreamer.json") as jsonFile:
        dreamer_results = json.load(jsonFile)
        jsonFile.close()

    plt.title(dreamer_results[60].get("task")+" seed: "+str(dreamer_results[60].get("seed")))
    plt.plot(dreamer_results[60].get("xs")[:10], dreamer_results[60].get("ys")[:10])
    plt.xticks(dreamer_results[60].get("xs")[:10])
    plt.show()

    for step, value in zip(dreamer_results[60].get("xs")[:10],dreamer_results[60].get("ys")[:10]):
        print("Return: {} for Step {}".format(value, step))




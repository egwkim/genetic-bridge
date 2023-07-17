import numpy as np
import math

from bridge import *

# TODO Set proper moment-stress coefficient.
ROAD_DENSITY = 500  # Road density relative to beam density
ROAD_COEFF = 1  # Moment-Stress coeff of road
BEAM_COEFF = 0  #  Moment-Stress coeff of beam

# TODO Set proper road fitenss coeff
# fitenss = max_road_stress * road_fitness_coeff + max_beam_stress
ROAD_FITENSS_COEFF = 0


INIT_SCALE = np.array([1.2, 3, 1.2, 3, 1.2, 3])
INIT_SHIFT = np.array([0, -1, 1, -1, 2, -1])


class Gene:
    def __init__(self, gene=None) -> None:
        if gene is None:
            self.gene = np.random.rand(6)
            self.gene *= INIT_SCALE
            self.gene += INIT_SHIFT
        else:
            self.gene = gene
        self._fitness = None

    @property
    def fitness(self):
        if self._fitness is not None:
            return self._fitness

        self.bridge = Bridge()
        self.bridge.pins = []
        self.bridge.beams = []
        self.bridge.pins.append(Pin(0, 0, True))
        self.bridge.pins.append(Pin(1, 0))
        self.bridge.pins.append(Pin(2, 0))
        self.bridge.pins.append(Pin(3, 0, True))
        self.bridge.pins.append(Pin(self.gene[0], self.gene[1]))
        self.bridge.pins.append(Pin(self.gene[2], self.gene[3]))
        self.bridge.pins.append(Pin(self.gene[4], self.gene[5]))

        self.bridge.beams.append(Beam(0, 1, ROAD_COEFF, ROAD_DENSITY))
        self.bridge.beams.append(Beam(1, 2, ROAD_COEFF, ROAD_DENSITY))
        self.bridge.beams.append(Beam(2, 3, ROAD_COEFF, ROAD_DENSITY))
        self.bridge.beams.append(Beam(0, 4, BEAM_COEFF))
        self.bridge.beams.append(Beam(1, 4, BEAM_COEFF))
        self.bridge.beams.append(Beam(1, 5, BEAM_COEFF))
        self.bridge.beams.append(Beam(2, 5, BEAM_COEFF))
        self.bridge.beams.append(Beam(2, 6, BEAM_COEFF))
        self.bridge.beams.append(Beam(3, 6, BEAM_COEFF))
        self.bridge.beams.append(Beam(4, 5, BEAM_COEFF))
        self.bridge.beams.append(Beam(5, 6, BEAM_COEFF))

        try:
            self.bridge.validate()
            self.bridge.solve()
            self.bridge.calculate_stress()
            max_stress = max(beam.stress for beam in self.bridge.beams[3:])
            max_road_stress = max(beam.stress for beam in self.bridge.beams[:3])
            self._fitness = -max_stress - ROAD_FITENSS_COEFF * max_road_stress

        except ValueError as e:
            self._fitness = -math.inf

        del self.bridge

        return self._fitness


class GeneticAlgorithm:
    def __init__(self, n) -> None:
        self.n = n
        self.population = [Gene() for i in range(n)]
        self.sorted = False

    def sort(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.sorted = True

    def next_generation(self):
        if not self.sorted:
            self.sort()

        # Preserve top population
        top = self.population[: self.n // 8]
        new = []
        new += top

        # Crossover
        for i in range(self.n // 3):
            parent1, parent2 = np.random.choice(top, 2, replace=False)
            gene = []
            # Prevent perfect copy of one parent
            rand = np.random.randint(1, 7)
            for j in range(3):
                if rand >> j & 1:
                    gene.append(parent1.gene[j * 2])
                    gene.append(parent1.gene[j * 2 + 1])
                else:
                    gene.append(parent2.gene[j * 2])
                    gene.append(parent2.gene[j * 2 + 1])
            # Append very small noise for diversity
            gene += (np.random.rand(6) - 0.5) * 0.05
            new.append(Gene(np.array(gene)))

        # Mutate
        # Small mutate of top gene
        gene = top[0].gene + (np.random.rand(6) - 0.5) * 0.05
        new.append(Gene(gene))
        for i in np.random.choice(len(top), self.n // 20):
            gene = top[i].gene + (np.random.rand(6) - 0.5) * 0.05
            new.append(Gene(gene))

        for i in np.random.choice(len(new), self.n - len(new)):
            # Small mutate (continuous mutate)
            gene = new[i].gene + (np.random.rand(6) - 0.5) * 0.1
            for j in range(len(gene)):
                # Large mutate with small chance
                if np.random.rand(1)[0] < 0.1:
                    gene[j] = np.random.rand(1)[0] * INIT_SCALE[j] + INIT_SHIFT[j]
            new.append(Gene(gene))

        self.population = new
        self.sorted = False


def main():
    ga = GeneticAlgorithm(200)
    for i in range(100):
        ga.next_generation()
        ga.sort()
        print(
            "Best: ",
            ga.population[0].fitness,
            "\tAvg: ",
            sum(i.fitness for i in ga.population) / ga.n,
        )
    print("\n=== Genetic algorithm finished ===")
    print("Top 10 genes: ")

    import matplotlib.pyplot as plt
    import os

    img_dir = "./pictures"
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    if not os.path.isdir("./pictures"):
        raise FileExistsError(img_dir + " is not a directory.")

    for i in range(10):
        print(ga.population[i].gene, ga.population[i].fitness)
        plt.plot([0, 1, 2, 3], [0, 0, 0, 0], "-o")
        plt.plot(
            ga.population[i].gene[::2],
            ga.population[i].gene[1::2],
            "-o",
        )
        plt.plot(
            [
                0,
                ga.population[i].gene[0],
                1,
                ga.population[i].gene[2],
                2,
                ga.population[i].gene[4],
                3,
            ],
            [
                0,
                ga.population[i].gene[1],
                0,
                ga.population[i].gene[3],
                0,
                ga.population[i].gene[5],
                0,
            ],
            "-",
        )
        plt.savefig(f"{img_dir}/{i}.png")
        plt.clf()


def test():
    gene = Gene([0.0080425, 0.79533181, 1.57538897, 0.44297026, 2.89131384, 0.00705552])
    print(gene.fitness())


if __name__ == "__main__":
    main()

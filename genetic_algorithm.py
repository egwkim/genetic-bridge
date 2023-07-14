import numpy as np
import math

from bridge import *


ROAD_DENSITY = 20
ROAD_COEFF = 1
BEAM_COEFF = 1


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
            max_stress = 0
            for beam in self.bridge.beams[3:]:
                max_stress = max(max_stress, beam.stress)
            self._fitness = -max_stress
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
        top = self.population[: self.n // 8]
        new = []
        new += top
        # Crossover
        for i in range(self.n // 3):
            parent1, parent2 = np.random.choice(top, 2)
            gene = []
            for j in range(3):
                if np.random.random(1)[0] < 0.5:
                    gene.append(parent1.gene[j * 2])
                    gene.append(parent1.gene[j * 2 + 1])
                else:
                    gene.append(parent2.gene[j * 2])
                    gene.append(parent2.gene[j * 2 + 1])
            new.append(Gene(np.array(gene)))
        # Mutate
        for i in np.random.choice(len(new), self.n // 4):
            gene = self.population[i].gene + (np.random.rand(6) - 0.5) * 0.2
            new.append(Gene(gene))
        for i in np.random.choice(len(new), self.n - len(new)):
            gene = self.population[i].gene
            for j in range(len(gene)):
                if np.random.rand(1)[0] < 0.1:
                    gene[j] = np.random.rand(1)[0] * INIT_SCALE[j] + INIT_SHIFT[j]
            new.append(Gene(gene))
        self.population = new
        self.sorted = False


def main():
    ga = GeneticAlgorithm(200)
    for i in range(500):
        ga.next_generation()
        ga.sort()
        print(ga.population[0].gene, ga.population[0].fitness)


if __name__ == "__main__":
    main()

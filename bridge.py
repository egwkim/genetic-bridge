import numpy as np

g = 1
MOMENT_STRESS_COEFF = 1


class Pin:
    """
    Pin connects multiple beams. It can convey force but not momentum.
    If a pin is on ground, it gets normal force from the ground
    Direction of the normal force is +y
    """

    def __init__(self, x, y, on_ground=False) -> None:
        self.pos = np.array((x, y))
        self.on_ground = on_ground
        self.beams = []
        self.force = 0


class Beam:
    def __init__(self, pin1, pin2, stress_coeff=1, density=1) -> None:
        self.pins = [None, None]
        self.pin_idx = [None, None]
        if isinstance(pin1, Pin):
            self.pins[0] = pin1
        else:
            self.pin_idx[0] = pin1
        if isinstance(pin2, Pin):
            self.pins[1] = pin2
        else:
            self.pin_idx[1] = pin2

        self.density = density
        self.stress_coeff = stress_coeff
        self.forces = None
        self._moment = None

    # Each beam should be updated only once.
    def update(self):
        self.pins[0].beams.append(self)
        self.pins[1].beams.append(self)
        self.vector = self.pins[0].pos - self.pins[1].pos
        self.length = np.linalg.norm(self.vector)
        if self.length == 0:
            raise ValueError("Bridge has zero length")
        self.direction = self.vector / self.length
        self.m = self.density * self.length

    @property
    def moment(self):
        # Relative value
        if self._moment is None:
            if self.forces is None:
                raise ValueError("Beam has no force.")
            self._moment = np.cross(self.vector, self.forces[0])
        return self._moment


class Bridge:
    """
    Bridge consists of multiple beams and pins.
    Solving a bridge means calculating all the forces when the birdge is in equilibrium.
    It also calculates maximum stress of each beam.

    If pins and beams are not given upon initialization,
    Three pins and two beams are created automatically.
    """

    def __init__(self, pins=[], beams=[]) -> None:
        self.pins = pins
        self.beams = beams

    def validate(self):
        for beam in self.beams:
            for i in [0, 1]:
                if beam.pins[i] == None:
                    beam.pins[i] = self.pins[beam.pin_idx[i]]
            beam.update()

        updated = False
        for i, pin in enumerate(self.pins[::-1]):
            if len(pin.beams) < 2:
                self.pins.pop(-i - 1)
                for beam in pin.beams:
                    try:
                        self.beams.remove(beam)
                        beam.pins[0].beams.remove(beam)
                        beam.pins[1].beams.remove(beam)
                    except ValueError:
                        pass
                updated = True
        if not updated:
            return

    def solve(self):
        # For each beam equilibrium of force, torque
        # Fore each pins equilibrium of force.
        n = len(self.beams)

        grounded_pins = []
        floating_pins = []
        for pin in self.pins:
            if pin.on_ground:
                grounded_pins.append(pin)
            else:
                floating_pins.append(pin)

        g_len = len(grounded_pins)
        f_len = len(floating_pins)
        if g_len - 1 + 2 * len(floating_pins) != n:
            # The number of components must satisfy the following
            # beams = 2 * (floating pins) + (grounded pins) - 1
            raise ValueError(f"Incorrect number of pins and beams. Beams: {n}, Grounded pins: {g_len}, Floating pins: {f_len}")

        matrix = np.zeros((4 * n, 4 * n))
        value = np.zeros(4 * n)

        # Equilibrium of force
        for i in range(n):
            matrix[2 * i][4 * i] = 1
            matrix[2 * i][4 * i + 2] = 1
            matrix[2 * i + 1][4 * i + 1] = 1
            matrix[2 * i + 1][4 * i + 3] = 1
            value[2 * i] = 0
            value[2 * i + 1] = self.beams[i].m * g

        # Equilibrium of torque
        for i in range(n):
            matrix[2 * n + i][4 * i] = self.beams[i].direction[1]
            matrix[2 * n + i][4 * i + 1] = -self.beams[i].direction[0]
            matrix[2 * n + i][4 * i + 2] = -self.beams[i].direction[1]
            matrix[2 * n + i][4 * i + 3] = +self.beams[i].direction[0]

        # Equilibrium of force on each pin
        # Omitting one equation about the x compenent
        # Since it is guaranteed by other euqtions.
        for i, beam in enumerate(self.beams):
            for j in [0, 1]:
                try:
                    k = grounded_pins.index(beam.pins[j])
                    if k == 0:
                        # Omitting one equation
                        continue
                    matrix[3 * n + k - 1][4 * i + 2 * j] = 1

                except ValueError:
                    k = floating_pins.index(beam.pins[j])
                    matrix[3 * n + g_len - 1 + 2 * k][4 * i + 2 * j] = 1
                    matrix[3 * n + g_len - 1 + 2 * k + 1][4 * i + 2 * j + 1] = 1

        self.matrix = matrix
        self.value = value
        self.forces = np.linalg.solve(matrix, value)
        # Force of ith beam is in index [4*i:4*i+4]
        # 4*ith force is the x component of the force
        # which is applied to the ith beam from it's first pin.
        # (4*i+1)th force is the y component of the same force.
        # (4*i+2)th force is the x component of the force
        # applied from the second pin.
        # (4*i+3)th force is the y component of the force
        # applied from the second pin.

        for i in range(n):
            self.beams[i].forces = [
                np.array((self.forces[4 * i], self.forces[4 * i + 1])),
                np.array((self.forces[4 * i + 2], self.forces[4 * i + 3])),
            ]

    def calculate_stress(self):
        for beam in self.beams:
            if beam.forces is None:
                raise ValueError("Beam has no force. Try solving first.")
            candidates = [
                beam.forces[0] + beam.direction * beam.stress_coeff * beam.moment,
                beam.forces[0] - beam.direction * beam.stress_coeff * beam.moment,
                beam.forces[1] + beam.direction * beam.stress_coeff * beam.moment,
                beam.forces[1] - beam.direction * beam.stress_coeff * beam.moment,
            ]
            beam.stress = max(np.linalg.norm(F) for F in candidates)


def main():
    bridge = Bridge()
    bridge.pins.append(Pin(0, 0, True))
    bridge.pins.append(Pin(1, 0))
    bridge.pins.append(Pin(2, 0, True))
    bridge.pins.append(Pin(0.5, 3**0.5 / 2))
    bridge.pins.append(Pin(1.5, 3**0.5 / 2))

    road_density = 20
    road_coeff = 1
    beam_coeff = 1

    bridge.beams.append(Beam(0, 1, road_coeff, road_density))
    bridge.beams.append(Beam(1, 2, road_coeff, road_density))
    bridge.beams.append(Beam(0, 3, beam_coeff))
    bridge.beams.append(Beam(1, 3, beam_coeff))
    bridge.beams.append(Beam(1, 4, beam_coeff))
    bridge.beams.append(Beam(2, 4, beam_coeff))
    bridge.beams.append(Beam(3, 4, beam_coeff))
    bridge.validate()
    print(
        f"The bridge consists of {len(bridge.pins)} pins and {len(bridge.beams)} beams"
    )
    # Print solution and value for debugging
    try:
        bridge.solve()
        print("Solution found")
    except ValueError as e:
        print("No solution")
        print(e)
    bridge.calculate_stress()
    for beam in bridge.beams:
        print(beam.forces, beam.m, beam.stress)
    print(bridge.value)


if __name__ == "__main__":
    main()

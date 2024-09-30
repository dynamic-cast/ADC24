class Mode:
    data_gathering = 0
    training = 1
    control = 2
    none = 3

class XYControl:
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        self._training_points = []
        self._mode = Mode.none

    def add_point(self, coordinates):
        if self._mode == Mode.data_gathering:
            print(f"Adding point: {coordinates}")
            self._training_points.append(coordinates)

    def toggle_data_gathering(self, value):
        if value == "on":
            self._mode = Mode.data_gathering
        elif value == "off":
            self._mode = Mode.none

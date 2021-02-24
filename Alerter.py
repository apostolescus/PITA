from globals import get_speed
from Storage import FrictionCoefficient, Constants


class Alerter:
    def __init__(
        self,
        car_type="standard_stock",
        weather_type="dry_asphalt",
        reaction_time=1,
    ):
        friction_coef = "FrictionCoefficient." + car_type + "." + weather_type
        friction_coef = eval(friction_coef)
        
        self.multiply = FrictionCoefficient.formula.multiplier / friction_coef
        self.reaction_time = reaction_time * Constants.km_to_h
        self.sound_duration = Constants.sound_duration
        self.sound_freq = Constants.sound_freq

        # print("friction coef: ", friction_coef)
        # print("multiply: ", self.multiply)

    def check_safety(self, distances):

        dictionary = distances.items()
        sorted_distances = sorted(dictionary)

        speed = get_speed()
        max_distance = self._calculate_max_distance(speed)

        for i in sorted_distances:
            if max_distance > i[0]:
                # make warning sound
                print(
                    "max distance: ", max_distance, "current distance: ", i[0]
                )

    # speed is in km/h
    def _calculate_max_distance(self, speed):
        d = (speed * speed) * self.multiply
        reaction_distance = speed * self.reaction_time

        return d + reaction_distance


# def test():
#     alert = Alerter()

#     for i in range(34,67):
#         d = alert._calculate_max_distance(i)
#         print("max distance is: ", d)

# test()



class FrictionCoefficient:

    class standard_stock:
        dry_asphalt = 1.3
        wet_asphalt = 0.8
        snow = 0.2
        ice = 0.1

    class truck:
        dry_asphalt = 0.8
        wet_asphalt = 0.55
        snow = 0.2
        ice = 0.1    

    class high_performance:
        dry_asphalt = 1
        wet_asphalt = 0.7
        snow = 0.15
        ice =  0.08      

    class tourism:
        dry_asphalt = 0.9
        wet_asphalt = 0.6
        snow = 0.2
        ice = 0.1
    
    class formula:
        multiplier = 0.003914
    
class Constants:
    km_to_h = 0.277
    sound_duration = 1000
    sound_freq = 440

class UISelected:

    car_type = 0 # 0 stock, 1 truck 2 bus 3 sport
    weather = 0 # 0 dry 1 wet 2 snow 3 ice
    experience = 0 #0 biginner 1 intermediate 2 advanced
    rec_mode = 0 # 0 smart mode 1 permanent 2 fix-size
    reaction_time = 0.5

    
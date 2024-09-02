import numpy as np
import matplotlib.pyplot as plt
from math import exp

class SplitACModel:
    def __init__(self, room_size, insulation_factor, ac_capacity, initial_temp, initial_humidity, thermal_mass, wall_conductance):
        self.room_size = room_size  # in square meters
        self.insulation_factor = insulation_factor  # 0 to 1, where 1 is perfect insulation
        self.ac_capacity = ac_capacity  # in BTUs per hour
        self.current_temp = initial_temp  # initial room temperature in °C
        self.current_humidity = initial_humidity  # initial room humidity in %
        self.setpoint_temp = initial_temp  # initial setpoint temperature in °C
        self.thermal_mass = thermal_mass  # in kJ/°C, represents the room's ability to store heat
        self.wall_conductance = wall_conductance  # in W/°C, represents heat transfer through walls
        self.air_velocity = 0.1  # Initial air velocity in m/s
        self.temp_history = []  # Track indoor temperature over time
        self.setpoint_history = []  # Track setpoint temperature over time
        self.outdoor_temp_history = []  # Track outdoor temperature over time
        self.outdoor_humidity_history = []  # Track outdoor humidity over time
        self.humidity_history = []  # Track indoor humidity over time
        self.pmv_history = []  # Track PMV over time
        self.air_velocity_history = []  # Track air velocity over time
        self.min_setpoint_temp = 16  # Minimum setpoint temperature for the AC
        self.max_setpoint_temp = 30  # Maximum setpoint temperature for the AC

    def calculate_heat_gain(self, outdoor_temp, time_hours):
        # Heat gain through conduction, factoring in insulation
        conduction_heat_gain = self.wall_conductance * (outdoor_temp - self.current_temp) * time_hours
        conduction_heat_gain *= (1 - self.insulation_factor)

        return conduction_heat_gain

    def update_temperature(self, outdoor_temp, time_hours):
        # Heat gain from outdoor temperature
        heat_gain = self.calculate_heat_gain(outdoor_temp, time_hours)
        
        # Cooling power of the AC
        cooling_power = (self.ac_capacity * 3.517 / 3600) * (self.current_temp - self.setpoint_temp) * time_hours  # in kW
        cooling_effect = cooling_power * time_hours
        
        # Temperature change calculation
        temp_change = (heat_gain - cooling_effect) / self.thermal_mass
        self.current_temp += temp_change

        # Store data for plotting
        self.temp_history.append(self.current_temp)
        self.setpoint_history.append(self.setpoint_temp)
        self.outdoor_temp_history.append(outdoor_temp)
        self.humidity_history.append(self.current_humidity)
        self.outdoor_humidity_history.append(self.current_humidity)
        self.air_velocity_history.append(self.air_velocity)

        # Calculate and store PMV
        self.pmv_history.append(self.calculate_pmv())

    def adjust_setpoint_and_air_velocity(self):
        # Adjust setpoint and air velocity to keep PMV within the range of -1 to 1
        pmv = self.calculate_pmv()
        target_pmv = 0  # We aim for a neutral comfort level
        k_p = 0.5  # Proportional control factor for setpoint, adjust as needed
        k_v = 0.1  # Proportional control factor for air velocity, adjust as needed

        # Adjust the setpoint temperature
        setpoint_adjustment = k_p * (target_pmv - pmv)
        self.setpoint_temp = max(self.min_setpoint_temp, min(self.max_setpoint_temp, self.setpoint_temp + setpoint_adjustment))

        # Adjust the air velocity
        if pmv > 1:
            self.air_velocity = min(self.air_velocity + k_v, 1.5)  # Increase air velocity up to a maximum of 1.5 m/s
        elif pmv < -1:
            self.air_velocity = max(self.air_velocity - k_v, 0.1)  # Decrease air velocity but not below 0.1 m/s

    def suggest_ac_action(self):
        if self.current_temp > self.setpoint_temp:
            return "Cooling"
        else:
            return "Standby"

    def simulate(self, outdoor_temp, forecasted_outdoor_temp_1, forecasted_humidity_1, time_hours):
        self.adjust_setpoint_and_air_velocity()
        self.update_temperature(outdoor_temp, time_hours)
        
        ac_action_1 = self.suggest_ac_action()
        
        # Print results
        print(f"Present Outside Temp: {outdoor_temp}°C, Present Outside Humidity: {self.current_humidity}%")
        print(f"Forecasted Outside Temp (+1 hour): {forecasted_outdoor_temp_1}°C, Forecasted Humidity (+1 hour): {forecasted_humidity_1}%")
        print(f"Current Inside Temp: {self.current_temp:.2f}°C, Current Inside Humidity: {self.current_humidity}%")
        print(f"AC Setpoint: {self.setpoint_temp}°C, Air Velocity: {self.air_velocity:.2f} m/s")
        print(f"Suggested AC Action for +1 hour: {ac_action_1}\n")

    def calculate_pmv(self, clo=0.5, met=1.2, air_velocity=None):
        """
        Calculate PMV (Predicted Mean Vote) using the indoor temperature and humidity.
        - clo: Clothing insulation (default: 0.5)
        - met: Metabolic rate (default: 1.2)
        - air_velocity: Air velocity in m/s (default: self.air_velocity)
        """
        if air_velocity is None:
            air_velocity = self.air_velocity

        ta = self.current_temp  # air temperature
        tr = ta  # mean radiant temperature (assumed to be equal to air temperature)
        rh = self.current_humidity  # relative humidity
        pa = rh * 10 * exp(16.6536 - 4030.183 / (ta + 235))  # partial water vapor pressure

        icl = 0.155 * clo  # thermal insulation of clothing in m2K/W
        m = met * 58.15  # metabolic rate in W/m2
        w = 0  # external work in W/m2 (assumed to be zero)
        mw = m - w  # internal heat production in the human body

        fcl = 1.05 + 0.1 * icl if icl > 0.078 else 1 + 0.2 * icl
        hcf = 12.1 * np.sqrt(air_velocity)
        taa = ta + 273
        tra = tr + 273
        tcl = taa + (35.5 - ta) / (3.5 * icl + 0.1)
        
        p1 = icl * fcl
        p2 = p1 * 3.96
        p3 = p1 * 100
        p4 = p1 * taa
        p5 = 308.7 - 0.028 * mw + p2 * ((tra / 100) ** 4)
        xn = tcl / 100
        xf = tcl / 50
        n = 0
        eps = 0.00015
        
        while abs(xn - xf) > eps:
            xf = (xf + xn) / 2
            hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
            hc = hcf if hcf > hcn else hcn
            xn = (p5 + p4 * hc - p2 * xf ** 4) / (100 + p3 * hc)
            n += 1
            if n > 150:
                break

        tcl = 100 * xn - 273

        hl1 = 3.05 * 0.001 * (5733 - 6.99 * mw - pa)
        hl2 = 0.42 * (mw - 58.15)
        hl3 = 1.7 * 0.00001 * m * (5867 - pa)
        hl4 = 0.0014 * m * (34 - ta)
        hl5 = 3.96 * fcl * (xn ** 4 - (tra / 100) ** 4)
        hl6 = fcl * hc * (tcl - ta)

        ts = 0.303 * exp(-0.036 * m) + 0.028
        pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)

        return pmv

# Generate a forecast profile for temperature and humidity
def generate_forecast_profile(hours):
    base_temp = 35  # Base outdoor temperature in °C
    base_humidity = 60  # Base outdoor humidity in %

    # Generate temperature and humidity profiles with some variation
    temp_profile = base_temp + 5 * np.sin(np.linspace(0, 2 * np.pi, hours)) + np.random.normal(0, 0.5, hours)
    humidity_profile = base_humidity + 10 * np.sin(np.linspace(0, 2 * np.pi, hours)) + np.random.normal(0, 1, hours)

    return temp_profile, humidity_profile

if __name__ == "__main__":
    ac_model = SplitACModel(
        room_size=15, 
        insulation_factor=0.7, 
        ac_capacity=5000, 
        initial_temp=30, 
        initial_humidity=50,
        thermal_mass=1000,  # Example thermal mass in kJ/°C
        wall_conductance=10  # Example wall conductance in W/°C
    )

    # Generate a 10-hour forecast profile
    forecasted_temps, forecasted_humidities = generate_forecast_profile(10)

    # Continuous simulation using the forecast profile
    for hour in range(9):  # Only go to hour 9 because we need the next hour's forecast
        outdoor_temp = forecasted_temps[hour]
        forecasted_temp_1 = forecasted_temps[hour + 1]
        forecasted_humidity_1 = forecasted_humidities[hour + 1]
        ac_model.simulate(outdoor_temp, forecasted_temp_1, forecasted_humidity_1, 1)

    # Plotting the results
    time_points = list(range(9))  # x-axis: time in hours (9 points)

    plt.figure(figsize=(12, 12))

    # Plot indoor and outdoor temperatures
    plt.subplot(5, 1, 1)
    plt.plot(time_points, ac_model.temp_history, label="Indoor Temperature (°C)", color="blue", marker='o')
    plt.plot(time_points, ac_model.outdoor_temp_history, label="Outdoor Temperature (°C)", color="orange", marker='o')
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.title("Indoor vs Outdoor Temperature")

    # Plot indoor and outdoor humidity
    plt.subplot(5, 1, 2)
    plt.plot(time_points, ac_model.humidity_history, label="Indoor Humidity (%)", color="green", marker='o')
    plt.plot(time_points, ac_model.outdoor_humidity_history, label="Outdoor Humidity (%)", color="purple", marker='o')
    plt.ylabel("Humidity (%)")
    plt.legend()
    plt.title("Indoor vs Outdoor Humidity")

    # Plot setpoint temperature
    plt.subplot(5, 1, 3)
    plt.plot(time_points, ac_model.setpoint_history, label="AC Setpoint (°C)", color="red", marker='o')
    plt.ylabel("Setpoint Temperature (°C)")
    plt.legend()
    plt.title("AC Setpoint Over Time")

    # Plot PMV (Thermal Comfort)
    plt.subplot(5, 1, 4)
    plt.plot(time_points, ac_model.pmv_history, label="PMV (Thermal Comfort)", color="brown", marker='o')
    plt.ylabel("PMV")
    plt.legend()
    plt.title("PMV (Thermal Comfort) Over Time")

    # Plot Air Velocity
    plt.subplot(5, 1, 5)
    plt.plot(time_points, ac_model.air_velocity_history, label="Air Velocity (m/s)", color="cyan", marker='o')
    plt.ylabel("Air Velocity (m/s)")
    plt.xlabel("Time (hours)")
    plt.legend()
    plt.title("Air Velocity Over Time")

    plt.tight_layout()
    plt.show()

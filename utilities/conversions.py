def convert_ft_to_m(feet):
    return feet * 0.3048

def convert_sqft_to_sqm(sqft):
    return sqft * 0.092903

def fan_speed_to_air_velocity(speed, ceiling_height, room_area):
    """
    Converts ceiling fan speed to air velocity in a room. We looked up the catalog from Polycabs: https://polycab.com/wp-content/uploads/2021/07/Fans-Range-Catalogue-2024.pdf
    Speed 1: 120 CMM
    Speed 2: 160 CMM
    Speed 3: 210 CMM
    Speed 4: 260 CMM
    Speed 5: 300 CMM
    
    Then, we convert cubic meters per minute (CMM) to cubic meters per second (m³/s): CMM/60=m³/s

    Assumptions: 
    Fan coverage area is 5 square meters
    Fan is mounted at the center of the room
    Air velocity is calculated at the center of the room at a height of 2.7 meters

    
    Parameters:
    speed (int): Fan speed (0 for Off, 1 to 5 for speed levels).
    ceiling_height (float): Height of the ceiling in feet. Default is 8.86 feet.
    room_area (float): Area of the room in square feet. Default is 215.28 square feet.
    
    Returns:
    float: Air velocity in meters per second.
    """
    
    # Convert inputs from feet to meters
    ceiling_height = convert_ft_to_m(ceiling_height)
    room_area = convert_sqft_to_sqm(room_area)
    
    # Check for valid speed input
    if speed < 0:
        speed = 0
    elif speed > 5:
        speed = 5
    
    air_delivery_cmm = {
        0: 0,
        1: 120,
        2: 160,
        3: 210,
        4: 260,
        5: 300
    }
    
    # Convert CMM to m³/s
    air_delivery_m3s = air_delivery_cmm[speed] / 60
    
    # Assume a typical fan coverage area of 5 m²
    coverage_area = 5
    
    # Calculate base air velocity in m/s
    base_velocity = air_delivery_m3s / coverage_area
    
    # Adjustments for ceiling height and room area
    height_factor = 2.7 / ceiling_height  # Normalized against typical height of 2.7 meters
    area_factor = 20 / room_area  # Normalized against typical room area of 20 square meters
    
    # Calculate adjusted air velocity
    adjusted_velocity = base_velocity * height_factor * area_factor
    
    return adjusted_velocity


# Example usage
# print(fan_speed_to_air_velocity(3))  # Default parameters in feet
# print(fan_speed_to_air_velocity(4, ceiling_height=9.84, room_area=269.10))
# print(fan_speed_to_air_velocity(2, ceiling_height=8.20, room_area=161.46))

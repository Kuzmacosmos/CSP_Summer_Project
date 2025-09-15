import numpy as np
import datetime as dt
"""
Stable Vers. 5.0

Code compiled by Billy Luo (2025), where the elliptical orbit estimation is 
adapted from Jenkins (2013). DOI: https://doi.org/10.1088/0143-0807/34/3/633

Some of the equations are adapted from other sources. The core equations can 
be referred to the final project report, with the corresponding citations made.

N.B. It seems that Jenkins does not consider the EoT effect on the SR/SS 
time calculations and azi/elv angles of the sun at a given day/time 
combination, while those equations work very well for a perfect Earth and a perfectly 
circular orbit shape. 

The EoT (equation of time) is the time difference between the apparent solar 
time (the "time" indicated by the Earth's rotation -> sun's path w.r.t the 
observation point) and the mean solar time (average solar path time by using 
24-hr convention). The EoT can shift the solar noon time by up to +/- 15 
mins, where the standard meridian can be 
shifted by a few degrees, where 4 minutes is equivalent to 1 meridian degree. 
Therefore, we need to compensate for such a time difference in terms of 
calculating the SR/SS times and solar azi/elv angles. 

NOAA's solar calculator 
https://gml.noaa.gov/grad/solcalc/ calculates those values considering the 
EoT effects, which match the displayed values of the mainstream astronomical 
observation software such as Stellarium.
"""

# CONSTANT List
omega = 2 * np.pi / 23.9345    # Earth's rotation rate, eq. (6) Jenkins [rad/hr]
epsilon = np.radians(23.44)          # Obliquity, eq. (3) Jenkins
alt_threshold = np.radians(-0.85)    # Sunrise/sunset altitude including atmospheric refraction (c.f. NOAA)
epoch = dt.datetime(2013, 1, 1, 0, 0) # t=0 epoch
equinox_2025 = dt.datetime(2025, 3, 20, 9, 1)  # Spring equinox 2025, UTC
lambda0_2025 = 132.96                 # Reference geo longitude for 2025, eq. (26)
# N.B. to calculate lambda0_2025, use eq. (26) of Jenkins:
# lambda0_2025 = (18 + (-7.16/60) - (9+1/60)) * 2pi/24 = 132.96 deg

# Helper functions
def days_since_epoch(dt_utc):
    """Return t in days since 2013-01-01 00:00Z."""
    return (dt_utc - epoch).total_seconds() / 86400

def mean_anomaly(t_days):
    """Mean anomaly M(t), eq. (13) of Jenkins."""
    return -0.0410 + 0.017202 * t_days

def ecliptic_longitude(t_days):
    """Ecliptic longitude phi(t), eq. (14) of Jenkins."""
    M = mean_anomaly(t_days)
    return -1.3411 + M + 0.0334 * np.sin(M) + 0.0003 * np.sin(2*M)

def solar_declination(t_days):
    """Solar declination delta, c.f. Eq. 3.1 of the final project report."""
    phi = ecliptic_longitude(t_days)
    return np.arcsin(np.sin(epsilon) * np.sin(phi))

def equation_of_time(day_number):
    """Approximate equation of time in minutes for day of year.
        c.f. Eq. 3.3 of the final project report.
    """
    B = 2 * np.pi * (day_number - 81) / 364
    return 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)

# Main functions
def solar_angles(lat_deg, lon_deg, dt_utc):
    """
    Compute solar elevation (rad) and geographic azimuth (deg clockwise from North)
    for given lat, lon, UTC datetime, using declination + equation-of-time.
    """
    # compute solar declination, i.e. delta
    t = days_since_epoch(dt_utc)
    delta = solar_declination(t)

    # equation of time (in minutes) and longitude correction (4 min per degree)
    day_number = dt_utc.timetuple().tm_yday
    eot = equation_of_time(day_number)  # in minutes
    time_offset = eot + 4 * lon_deg  # minutes, c.f. Eq. 3.4 of the final report

    # apparent solar time in minutes from local midnight
    tst = dt_utc.hour * 60 + dt_utc.minute + dt_utc.second / 60 + time_offset

    # hour angle H (radians), zero at solar noon, positive in the afternoon
    # c.f. Eq. 3.5 of the final report
    H = np.radians(tst / 4.0 - 180.0)

    # convert latitude to radians
    L = np.radians(lat_deg)

    # elevation (altitude) above horizon, c.f. Eq. 3.6 from the final report
    elevation = np.arcsin(np.sin(L) * np.sin(delta) +
        np.cos(L) * np.cos(delta) * np.cos(H))

    delta_deg = np.degrees(delta)
    eps = 0.05  # tolerance in degrees for “zenith‐crossing day”
    # if today’s declination is (almost) equal to our latitude...
    if abs(delta_deg - lat_deg) < eps:
        # at the true‐solar‐noon instant H=0 approx the Sun is exactly overhead
        # so enforce the  E→S→W jump:
        if H < 0: az_deg = 90.0
        elif H > 0: az_deg = 270.0
        else: az_deg = 180.0
        return np.rad2deg(elevation), az_deg
    # azimuth: clockwise from North, c.f. Eq. 3.7 from the final report
    az_rad = np.arctan2(-np.sin(H),
        np.tan(delta) * np.cos(L) - np.sin(L) * np.cos(H))
    azimuth = (np.degrees(az_rad) + 360) % 360

    return np.rad2deg(elevation), azimuth

def compute_sunrise_sunset(lat_deg, lon_deg, date_local, tz_offset_hours=0):
    """
    Compute local sunrise and sunset times accounting for date, location, and timezone.
    Returns (sunrise_local, sunset_local), or (None, None) for polar day/night.
    c.f. eq. (5) from Jenkins.
    """
    # UTC midnight for the local date
    utc_midnight = dt.datetime(date_local.year, date_local.month, date_local.day) \
                  - dt.timedelta(hours=tz_offset_hours)
    # declination at midnight
    t = days_since_epoch(utc_midnight)
    decl = solar_declination(t)
    L_rad = np.radians(lat_deg)

    # compute the cosine of the sunrise hour angle
    cos_H0 = (np.sin(alt_threshold) - np.sin(L_rad)*np.sin(decl)) \
             / (np.cos(L_rad)*np.cos(decl))
    # polar day/night check
    if cos_H0 < -1 or cos_H0 > 1:
        return None, None

    # actual hour angle at sunrise/sunset (radians)
    H0 = np.arccos(cos_H0)
    # half-day length in hours
    half_day = H0 / (2 * np.pi) * 24

    # standard-meridian and longitude shift (hours)
    std_meridian = tz_offset_hours * 15
    lon_corr_hours = (std_meridian - lon_deg) / 15.0

    # equation of time effect (minutes) w.r.t. hours
    day_num = utc_midnight.timetuple().tm_yday
    eot_min  = equation_of_time(day_num)
    eot_hours = eot_min / 60.0

    # solar noon w.r.t. MEAN solar time, local TZ
    solar_noon_local = 12.0 + lon_corr_hours - eot_hours

    # assemble the date-times
    base = dt.datetime(date_local.year, date_local.month, date_local.day)
    sunrise_local = base + dt.timedelta(hours=solar_noon_local - half_day)
    sunset_local  = base + dt.timedelta(hours=solar_noon_local + half_day)

    return sunrise_local, sunset_local

# Formatting outputs
def format_timezone(offset_hours):
    sign = '+' if offset_hours >= 0 else '-'
    total_minutes = abs(int(offset_hours * 60))
    hh = total_minutes // 60
    mm = total_minutes % 60
    return f"UTC{sign}{hh:02d}:{mm:02d}"

def format_lat_lon(lat_deg, lon_deg):
    lat_dir = 'N' if lat_deg >= 0 else 'S'
    lon_dir = 'E' if lon_deg >= 0 else 'W'
    return f"{abs(lat_deg):.2f}°{lat_dir}", f"{abs(lon_deg):.2f}°{lon_dir}"
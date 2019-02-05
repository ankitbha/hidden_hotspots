Kaiterra data time window = 11 May - 10 June, 2018
==================================================

Kaiterra_11May-10June_loc.csv
-----------------------------

18 rows = 18 locations, 5 columns:
  - col 1 = character string for field_egg_id (= unique location identifier)
  - col 2 = numerical for longitude
  - col 3 = numerical for latitude
  - col 4 = numerical for projected UTM X
  - col 5 = numerical for projected UTM Y


Kaiterra_11May-10June_pm25.csv
------------------------------

2881 rows = time points, 19 columns = PM2.5 measurements for 18 locations:
  - col 1: character string for time stamp
  - col 2-19: numerical for pm25, one for each field_egg_id


Kaiterra_11May-10June_weather.csv
---------------------------------

2881 rows = time points but original data only hourly so each original values
repeated 4 times (except at boundaries and missing June 1st at 7am UTC).

Coarse spatial resolution: variables not geo-referenced so for the whole Delhi.

4 columns for some selected weather information:
  - col 1: time stamp, matching original time info but in IST
  - col 2: temperature converted to degrees celsius
  - col 3: humidity (relative?)
  - col 4: windspeed (unit?)


Kaiterra_11May-10June_coord.csv
-------------------------------

54 rows = 18 locations are which we have sensors + 36 extra locations obtained
by constrained Delaunay triangulation within a projected rectangle with corners:

|   |     utmx |     utmy |
|---|----------|----------|
| 1 | 699.1732 | 3143.883 |
| 2 | 733.1196 | 3143.883 |
| 3 | 733.1196 | 3170.463 |
| 4 | 699.1732 | 3170.463 |

3 columns:
  - col 1 : pseudo mercator UTM coordinates, easting (X)
  - col 2 : pseudo mercator UTM coordinates, northing (Y)
  - col 3 : 0-1 indicator to identify extra locations from original ones. Data
            are order so that the first 18 rows are the original ones (=1).



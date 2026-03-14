# The original dataset is from https://simplemaps.com/data/world-cities
# It is under the Creative Commons Attribution 4.0 license.
import string
ascii_lower_set = set(string.ascii_lowercase)

cities = set()
with open('worldcities.csv', 'r', encoding='utf-8') as f:
    for line in f.readlines()[1:]:
        city_raw = line.split(',')[1].strip('"') # The city_ascii col, with the quotes removed
        # I only want cities that do not contain spaces, and only conatin latin letters,
        # with only lower case letters.
        city = city_raw.lower()
        if set(city).issubset(ascii_lower_set):
            cities.add(city)

cities.remove('')

with open('cities_normalized.txt', 'w') as f:
    f.write('\n'.join(cities))
import json
import csv
import pandas as pd
import numpy as np
import unicodedata
import ast
from tqdm import tqdm

def parse_data(source1,output_path):
    #setup an array for writing each row in the csv file
	rows = []
    # extract fields from business json data set #
    # setup an array for storing each json entry
	business_data = []
    # setup an array for headers we are not using strictly
	business_header_removals = ['attributes', 'categories', 'hours', 'neighborhood', 'is_open','latitude','longitude','address']
    # setup an array for headers we are adding
    # business_header_additions = ['Noise Level', 'Attire', 'Alcohol', 'Price_Range', 'Delivery', 'Outdoor_Seating',
    #                              'Drive-Thru', 'Good_for_Groups', 'Has_TV', 'Caters', 'Waiter_Service',
    #                              'Good_for_Kids', 'Accepts_Credit_Cards', 'Takes_Reservations', 'Wi_Fi', 'Happy_Hour',
    #                              'Good_for_Dancing', 'Smoking', 'BYOB', 'Corkage', 'Take_Out', 'Coat_Check',
    #                              'Parking_Street', 'Parking_Valet', 'Parking_Lot', 'Parking_Garage',
    #                              'Parking_Validated', 'Music_DJ', 'Music_Karaoke', 'Music_Video', 'Music_Live',
    #                              'Music_Jukebox', 'Music_Background_Music', 'Is_Restaurants', 'Sandwiches', 'Fast Food',
    #                              'Nightlife', 'Pizza', 'Bars', 'Mexican', 'Food', 'American (Traditional)',
    #                              'Burgers', 'Chinese', 'Italian', 'American (New)', 'Breakfast & Brunch', 'Thai',
    #                              'Indian', 'Sushi Bars', 'Korean', 'Mediterranean', 'Japanese', 'Seafood',
    #                              'Middle Eastern', 'Pakistani', 'Barbeque', 'Vietnamese', 'Asian Fusion', 'Diners',
    #                              'Greek', 'Vegetarian']
    
	
    # open the business source file
	with open(source1) as f:	
		# for each line in the json file
		for line in f:
			# store the line in the array for manipulation
			business_data.append(json.loads(line))
	# close the reader
	f.close()

	print business_data[1].keys()

    #append the initial keys as csv headers
	header = sorted(business_data[0].keys())

	# remove keys from the business data that we are not using strictly
	for headers in business_header_removals:
		header.remove(headers)

	print header    
    # append the additional business related csv headers
    #for headers in business_header_additions:
    #    header.append(headers)

	print('processing data in the business dataset...')
    # for every entry in the business data array
	headerRow = ['business_id','address','city','state','postal_code','name','review_count','stars','type','BikeParking','BusinessAcceptsBitcoin',
				'BusinessAcceptsCreditCards',
				'BusinessParking_garage','BusinessParking_street','BusinessParking_validated','BusinessParking_lot','BusinessParking_valet',
				'RestaurantsPriceRange','WheelchairAccessible']
	modified_data = []
	#for entry in tqdm(range(0, len(business_data))):
	for entry in tqdm(range(0, 50)):    	
		row = []    
		row.append(business_data[entry]['business_id'])
		row.append(business_data[entry]['address'])
		row.append(business_data[entry]['city'])
		row.append(business_data[entry]['state'])
		row.append(business_data[entry]['postal_code'])
		row.append(business_data[entry]['name'])
		row.append(business_data[entry]['review_count'])
		row.append(business_data[entry]['stars'])
		row.append(business_data[entry]['type'])

		if(business_data[entry]['attributes'] is not None):
			attributesList = business_data[entry]['attributes']
			if('BikeParking: True' in business_data[entry]['attributes']):
				row.append(1)
			else:
				row.append(0)

			if('BusinessAcceptsBitcoin: True' in business_data[entry]['attributes']):
				row.append(1)
			else:
				row.append(0)

			if('BusinessAcceptsCreditCards: True' in business_data[entry]['attributes']):
				row.append(1)
			else:
				row.append(0)	
			
			businessParkingFound = False
			RestaurantsPriceRangeFound = False
			for element in business_data[entry]['attributes']:
				if('BusinessParking' in element):
					businessParkingFound = True
					# To convert this string into a dictionary we use ast.literal_eval
					parkingDict = ast.literal_eval(element[len('BusinessParking: '):])

					if ('garage' in parkingDict.keys() and parkingDict['garage']):
						row.append(1)
					else:
						row.append(0)

					if ('street' in parkingDict.keys() and parkingDict['street']):
						row.append(1)
					else:
						row.append(0)

					if ('validated' in parkingDict.keys() and parkingDict['validated']):
						row.append(1)
					else:
						row.append(0)

					if ('lot' in parkingDict.keys() and parkingDict['lot']):
						row.append(1)
					else:
						row.append(0)

					if ('valet' in parkingDict.keys() and parkingDict['valet']):
						row.append(1)
					else:
						row.append(0)
			
				if('RestaurantsPriceRange2' in element):
					RestaurantsPriceRangeFound = True
					row.append(element[len('RestaurantsPriceRange2 '):])

			# If BusinessParking attribute is not present in the file, then we assume there is no business parking	
			if(businessParkingFound == False):
				row.append(0)
				row.append(0)
				row.append(0)
				row.append(0)
				row.append(0)

			if(RestaurantsPriceRangeFound == False):
				row.append(0)

			if('WheelchairAccessible: True' in attributesList):
				row.append(1)
			else:
				row.append(0)			

		else:
			row.append(0)
			row.append(0)
			row.append(0)
			row.append(0)
			row.append(0)
			row.append(0)
			row.append(0)
			row.append(0)
			row.append(0)
			row.append(0)


		modified_data.append(row)
	
	
	#write to csv file
	#print len(modified_data)
	with open(output_path, 'w') as out:
		writer = csv.writer(out)
		# write the csv headers
		writer.writerow(headerRow)
		# for each entry in the row array
		print('writing contents to csv...')
		for entry in tqdm(range(0, len(modified_data))):
			try:
				# write the row to the csv
				writer.writerow(modified_data[entry])
			# if there is an error, continue to the next row
			except UnicodeEncodeError:
				continue
	out.close()
	








parse_data('yelp_academic_dataset_business.json','test.csv')
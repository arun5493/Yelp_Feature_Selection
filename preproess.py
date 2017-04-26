import json
import csv
import pandas as pd
import numpy as np
import unicodedata
import ast
from tqdm import tqdm
import operator

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
			#print "inside line"
			# store the line in the array for manipulation
			business_data.append(json.loads(line))
	# close the reader
	f.close()

	print business_data[1].keys()

    #append the initial keys as csv headers
	header = sorted(business_data[0].keys())
	allCategoriesList = []
	for entry in tqdm(range(0, len(business_data))):
		categoryList = [] 
		categoryList.append(business_data[entry]['categories'])
		#print categoryList[0]
		if categoryList[0] is not None:
			for each in categoryList[0]:
				allCategoriesList.append(each)
				
	categoriesCount = {}
	for each in allCategoriesList:
		if each not in categoriesCount.keys():
			categoriesCount[each] = 0
		categoriesCount[each]+=1
	sorted_x = sorted(categoriesCount.items(), key=operator.itemgetter(1))
	print sorted_x

	# remove keys from the business data that we are not using strictly
	for headers in business_header_removals:
		header.remove(headers)

	print('processing data in the business dataset...')
    # for every entry in the business data array
	headerRow = ['business_id','address','city','state','postal_code','name','review_count','stars','type','BikeParking','BusinessAcceptsBitcoin',
				'BusinessAcceptsCreditCards',
				'BusinessParking_garage','BusinessParking_street','BusinessParking_validated','BusinessParking_lot','BusinessParking_valet',
				'RestaurantsPriceRange',
				'Ambience_romantic','Ambience_intimate','Ambience_classy','Ambience_hipster','Ambience_divey','Ambience_touristy','Ambience_trendy','Ambience_upscale','Ambience_casual',
				'WheelchairAccessible','Alcohol','Caters','GoodForKids','HasTV','OutdoorSeating','RestaurantsDelivery','RestaurantsGoodForGroups',
				'RestaurantsReservations','RestaurantsTakeOut','RestaurantsTableService','WiFi',
				'Music_DJ','Music_Background','Music_Karaoke','Music_Live','Music_Video','Music_Jukebox']

	businessParkingList = ['garage','street','validated','lot','valet']			
	ambienceList = ['romantic','intimate','classy','hipster','divey','touristy','trendy','upscale','casual']
	musicList = ['dj','background_music','karaoke','live','video','jukebox']
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

		# if business_data[entry]['business_id'] == 'CGliHrLYH8ABT3k19kLF3w':	#This Data is not added to CSV. Need to check why
		# 	print business_data[entry]

		if(business_data[entry]['attributes'] is not None):
			attributesList = business_data[entry]['attributes']
			if('BikeParking: True' in business_data[entry]['attributes']):
				row.append(1)
			else:
				row.append(0)

			if('BusinessAcceptsBitcoin: True' in attributesList):
				row.append(1)
			else:
				row.append(0)

			if('BusinessAcceptsCreditCards: True' in attributesList):
				row.append(1)
			else:
				row.append(0)	
			
			businessParkingFound = False
			RestaurantsPriceRangeFound = False
			ambienceFound = False
			musicFound = False
			for element in attributesList:
				if('BusinessParking' in element):
					businessParkingFound = True
					# To convert this string into a dictionary we use ast.literal_eval
					parkingDict = ast.literal_eval(element[len('BusinessParking: '):])

					for each in businessParkingList:
						if each in parkingDict.keys() and parkingDict[each]:
							row.append(1)
						else:
							row.append(0)
			# If BusinessParking attribute is not present in the file, then we assume there is no business parking	
			if(businessParkingFound == False):
				row.append(0)
				row.append(0)
				row.append(0)
				row.append(0)
				row.append(0)					
			
			for element in attributesList:
				if('RestaurantsPriceRange2' in element):
					RestaurantsPriceRangeFound = True
					row.append(element[len('RestaurantsPriceRange2 '):])		# Watch out for its type.
			if(RestaurantsPriceRangeFound == False):
				row.append(0)

			for element in attributesList:		
				if('Ambience' in element):
					ambienceFound = True
					# To convert this string into a dictionary we use ast.literal_eval
					ambienceDict = ast.literal_eval(element[len('Ambience: '):])
					#print ambienceDict
					for each in ambienceList:
						if each in ambienceDict.keys() and ambienceDict[each]:
							row.append(1)
						else:
							row.append(0)

			if(ambienceFound == False):
				row.append(0)
				row.append(0)
				row.append(0)
				row.append(0)
				row.append(0)
				row.append(0)
				row.append(0)
				row.append(0)
				row.append(0)
			
			if('WheelchairAccessible: True' in attributesList):
				row.append(1)
			else:
				row.append(0)

			if('Alcohol: full_bar' in attributesList):
				row.append(2)
			elif('Alcohol: beer_and_wine' in attributesList):
				row.append(1)
			else:
				row.append(0)

			if('Caters: True' in attributesList):
				row.append(1)
			else:
				row.append(0)

			if('GoodForKids: True' in attributesList):
				row.append(1)
			else:
				row.append(0)	

			if('HasTV: True' in attributesList):
				row.append(1)
			else:
				row.append(0)

			if('OutdoorSeating: True' in attributesList):
				row.append(1)
			else:
				row.append(0)							

			if('RestaurantsDelivery: True' in attributesList):
				row.append(1)
			else:
				row.append(0)

			if('RestaurantsGoodForGroups: True' in attributesList):
				row.append(1)
			else:
				row.append(0)

			if('RestaurantsReservations: True' in attributesList):
				row.append(1)
			else:
				row.append(0)

			if('RestaurantsTakeOut: True' in attributesList):
				row.append(1)
			else:
				row.append(0)	

			if('RestaurantsTableService: True' in attributesList):
				row.append(1)
			else:
				row.append(0)			
			
			if('WiFi: free' in attributesList):
				row.append(2)
			elif ('WiFi: paid' in attributesList):
				row.append(1)
			else:
				row.append(0)

			for element in attributesList:		
				if('Music: ' in element):
					musicFound = True
					# To convert this string into a dictionary we use ast.literal_eval
					musicDict = ast.literal_eval(element[len('Music: '):])
					#print musicDict
					for each in musicList:
						if each in musicDict.keys() and musicDict[each]:
							row.append(1)
						else:
							row.append(0)

			if(musicFound == False):
				row.append(0)
				row.append(0)
				row.append(0)
				row.append(0)
				row.append(0)
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
			row.append(0)		# For Alcohol
			row.append(0)		# For Ambience
			row.append(0)
			row.append(0)
			row.append(0)
			row.append(0)
			row.append(0)
			row.append(0)
			row.append(0)
			row.append(0)
			row.append(0)		# For Cater
			row.append(0)		# For GoodForKids
			row.append(0)		# For HasTV
			row.append(0)		# For OutdoorSeating
			row.append(0)		# Restaurant Delivery
			row.append(0)		# Restaurant Groups
			row.append(0)		# Reservations
			row.append(0)		# TakeOuts
			row.append(0)		# TableServices
			row.append(0)		# Wifi
			row.append(0)		# For Music
			row.append(0)
			row.append(0)
			row.append(0)
			row.append(0)
			row.append(0)

		modified_data.append(row)
	
	
	#write to csv file
	print len(modified_data)
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
				#print "There is an error now"
				continue
	out.close()
	








parse_data('yelp_academic_dataset_business.json','test.csv')
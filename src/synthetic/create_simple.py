from mido import Message, MetaMessage, MidiFile, MidiTrack
from utils import *
import numpy as np
from random import randint
import os
import math

from midi2audio import FluidSynth
import os
# store sound fonts in this dir ~/.fluidsynth/default_sound_font.sf2
sound_font = "gs.sf2"


fs = FluidSynth( sound_font, sample_rate = 44100 )

def get_sample_index_from_tick( cur_tick, tick_per_beat, tempo, sample_rate ):
	beat = cur_tick / tick_per_beat
	ms = beat * tempo 
	return math.ceil( ms / 1000000 * sample_rate )

def create_gaussian( array, position, window_resolution = 10 ):
	pass

# length in miliseconds
# tempo 500000 is 120 beats per minute
def create_midi_file( file_name, length, instrument = 1,
	                                     _range = ( 21, 108 ),
	                                     simul = 1,
	                                     key_signature = "C", tempo = 500000, sample_rate = 44100, 
	                                     include_offset = True, ticks_per_beat = 480 ):

	
	note_floor, note_ceil = get_instrument_note_range( instrument )

	mid = MidiFile()
	mid.ticks_per_beat = ticks_per_beat
	track = MidiTrack()
	mid.tracks.append(track)
	num_sample = math.ceil( length / 1000000 * sample_rate )
	num_beat = length / tempo
	num_tick = math.floor( mid.ticks_per_beat * num_beat )

	# assert( num_tick % num_sample == 0, "sample not divisible by ticks" )
	label = np.zeros( num_sample, dtype = np.float32 )
	"""
	ALL IN TICKS
	"""
	# setup metadata and control
	track.append( MidoMessage.track_name( name = file_name ) )
	track.append( MidoMessage.key_signature( key = key_signature ) )
	track.append( MidoMessage.set_tempo( tempo = tempo ) )
	track.append( MidoMessage.time_signature() )
	track.append( MidoMessage.program_change( chan = 0, prog = instrument, dt = 0 ) )
	track.append( MidoMessage.control_change( chan = 0, control = 64, value = 0, dt = 0 ) )
	track.append( MidoMessage.control_change( chan = 0, control = 91, value = 48, dt = 0 ) )
	track.append( MidoMessage.control_change( chan = 0, control = 10, value = 51, dt = 0 ) )
	track.append( MidoMessage.control_change( chan = 0, control = 7, value = 100, dt = 0 ) )
	
	portion = math.floor( num_tick / 8 )

	start_tick = randint( portion, 2*portion )
	sound_length = randint( 2*portion, 5*portion )
	lower, upper = _range
	# if single note
	if simul == 1:
		note = randint( lower, upper )
		track.append( MidoMessage.note_on( chan = 0, note = note, vel = 100, dt = start_tick ) )
		track.append( MidoMessage.note_off( chan = 0, note = note, vel = 100, dt =  sound_length ) )
	else:
		# if multi note
		num_note = randint( 2, simul )
		note_list = []
		for i in range( num_note ):
			note = randint( lower, upper )
			note_list.append( note )
		note_start = True
		for note in note_list:
			if note_start:
				track.append( MidoMessage.note_on( chan = 0, note = note, vel = 100, dt = start_tick ) )
				note_start = False
			else:
				track.append( MidoMessage.note_on( chan = 0, note = note, vel = 100, dt = 0 ) )
		note_off = True
		for note in note_list:
			if note_off:
				track.append( MidoMessage.note_off( chan = 0, note = note, vel = 100, dt = sound_length ) )
				note_off = False
			else:
				track.append( MidoMessage.note_off( chan = 0, note = note, vel = 100, dt = 0 ) )
				
	label[ get_sample_index_from_tick( start_tick, mid.ticks_per_beat, tempo, sample_rate ) ] = 1.
	if include_offset:
		label[ get_sample_index_from_tick( start_tick + sound_length, mid.ticks_per_beat, tempo, sample_rate ) ] = 1.

	return mid, label

name_dict = { 1:"piano", 41:"violin", 43:"cello", 47:"harp", 26:"acoustic_guitar_steel", 65:"oprano_sax" }
range_dict = { 1:(21,108), 41:(55,105), 43:(36,84), 47:(24,100), 26:(52,88), 65:(59,91) }
skip_list = []
os.mkdir( "train" ) if not os.path.isdir( "train" ) else print()
for instrument_index, name in name_dict.items():
	try:
		instrument = name
		_range = range_dict[ instrument_index ]
		base_dir = "train/" + instrument
		os.mkdir( base_dir ) if not os.path.isdir( base_dir ) else print()

		single_dir = base_dir + "/" + "single"
		os.mkdir( single_dir ) if not os.path.isdir( single_dir ) else print()
		sound_folder = single_dir + "/" + "mono_sound"
		label_folder = single_dir + "/" + "mono_label"
		midi_folder = single_dir + "/" + "mono_midi"
		os.mkdir( sound_folder ) if not os.path.isdir( sound_folder ) else print()
		os.mkdir( label_folder ) if not os.path.isdir( label_folder ) else print()
		os.mkdir( midi_folder ) if not os.path.isdir( midi_folder ) else print()

		for i in range( 1000 ):

			file_name = instrument + "_mono_" + str( i )
			mid, arr = create_midi_file( file_name, length = 6000000, instrument = instrument_index, _range = _range, 
				                         tempo = 441000, include_offset = False )
			mid.save( midi_folder + "/" + file_name + '.mid' )
			fs.midi_to_audio( midi_folder + '/' + file_name + '.mid', sound_folder + "/" + file_name + ".wav" )
			np.save( label_folder + "/" + file_name + ".npy", arr )

		# double_dir = base_dir + "/" + "double"
		# os.mkdir( double_dir ) if not os.path.isdir( double_dir ) else print()
		# sound_folder = double_dir + "/" + "mono_sound"
		# label_folder = double_dir + "/" + "mono_label"
		# midi_folder = double_dir + "/" + "mono_midi"
		# os.mkdir( sound_folder ) if not os.path.isdir( sound_folder ) else print()
		# os.mkdir( label_folder ) if not os.path.isdir( label_folder ) else print()
		# os.mkdir( midi_folder ) if not os.path.isdir( midi_folder ) else print()

		
		# for i in range( 1000 ):

		# 	file_name = instrument + "_mono_" + str( i )
		# 	mid, arr = create_midi_file( file_name, length = 6000000, instrument = instrument_index, _range = _range, 
		# 		                         simul = 5, tempo = 441000, include_offset = False )
		# 	mid.save( midi_folder + "/" + file_name + '.mid' )
		# 	fs.midi_to_audio( midi_folder + '/' + file_name + '.mid', sound_folder + "/" + file_name + ".wav" )
		# 	np.save( label_folder + "/" + file_name + ".npy", arr )
	except:
		skip_list.append( name )
print( "Skipping instrument", skip_list )
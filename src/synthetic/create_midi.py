from mido import Message, MetaMessage, MidiFile, MidiTrack
from utils import *
import numpy as np
from random import randint
import os
import math

def get_sample_index_from_tick( cur_tick, tick_per_beat, tempo, sample_rate ):
	beat = cur_tick / tick_per_beat
	ms = beat * tempo 
	return math.ceil( ms / 1000000 * sample_rate )

# length in miliseconds
# tempo 500000 is 120 beats per minute
def create_midi_file( file_name, length, instrument = 1,key_signature = "C", tempo = 500000, sample_rate = 44100, include_offset = True, ticks_per_beat = 480 ):

	
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
	
	prev_rest = 0
	total_tick = 0
	while num_tick > 10:

		# note
		note = randint( note_floor, note_ceil )
		time = randint( 10, min( num_tick ,600 ) )
		track.append( MidoMessage.note_on( chan = 0, note = note, vel = 100, dt = prev_rest ) )
		track.append( MidoMessage.note_off( chan = 0, note = note, vel = 100, dt = prev_rest + time ) )
		num_tick -= time 
		total_tick += prev_rest 
		label[ get_sample_index_from_tick( total_tick, mid.ticks_per_beat, tempo, sample_rate ) ] = 1.
		total_tick += time
		if( include_offset ):
			label[ get_sample_index_from_tick( total_tick, mid.ticks_per_beat, tempo, sample_rate ) ] = 1.
			
		# rest
		prev_rest = randint( 0, min( num_tick ,200 ) )
		num_tick -= prev_rest

	return mid, label

os.mkdir( "midi_sax" )
os.mkdir( "label_sax" )
for i in range( 100 ):

	file_name = "piano_single_" + str( i )
	mid, arr = create_midi_file( file_name, length = 8000000, instrument = 65, tempo = 441000 )

	
	mid.save( "midi_sax/" + file_name + '.mid' )
	np.save( "label_sax/" + file_name + "", arr )
from typing import List, Optional
import subprocess

import config.globals as globals
from config.type import OutputVideoPreset, Fps
import utils.logger as logger
from utils.filesystem import get_temp_frames_pattern, get_temp_output_video_path


def run_ffmpeg(args : List[str]) -> bool:
	commands = [ 'ffmpeg', '-hide_banner', '-loglevel', 'error' ]
	commands.extend(args)
	try:
		subprocess.run(commands, stderr = subprocess.PIPE, check = True)
		return True
	except subprocess.CalledProcessError as exception:
		logger.debug(exception.stderr.decode().strip(), __name__.upper())
		return False


def open_ffmpeg(args : List[str]) -> subprocess.Popen[bytes]:
	commands = [ 'ffmpeg', '-hide_banner', '-loglevel', 'error' ]
	commands.extend(args)
	return subprocess.Popen(commands, stdin = subprocess.PIPE)


def extract_frames(target_path : str, video_resolution : str, video_fps : Fps) -> bool:
	temp_frame_compression = round(31 - (globals.temp_frame_quality * 0.31))
	trim_frame_start = globals.trim_frame_start
	trim_frame_end = globals.trim_frame_end
	temp_frames_pattern = get_temp_frames_pattern(target_path, '%04d')
	commands = [ '-hwaccel', 'auto', '-i', target_path, '-q:v', str(temp_frame_compression), '-pix_fmt', 'rgb24' ]
	if trim_frame_start is not None and trim_frame_end is not None:
		commands.extend([ '-vf', 'trim=start_frame=' + str(trim_frame_start) + ':end_frame=' + str(trim_frame_end) + ',scale=' + str(video_resolution) + ',fps=' + str(video_fps) ])
	elif trim_frame_start is not None:
		commands.extend([ '-vf', 'trim=start_frame=' + str(trim_frame_start) + ',scale=' + str(video_resolution) + ',fps=' + str(video_fps) ])
	elif trim_frame_end is not None:
		commands.extend([ '-vf', 'trim=end_frame=' + str(trim_frame_end) + ',scale=' + str(video_resolution) + ',fps=' + str(video_fps) ])
	else:
		commands.extend([ '-vf', 'scale=' + str(video_resolution) + ',fps=' + str(video_fps) ])
	commands.extend([ '-vsync', '0', temp_frames_pattern ])
	return run_ffmpeg(commands)


def compress_image(output_path : str) -> bool:
	output_image_compression = round(31 - (globals.output_image_quality * 0.31))
	commands = [ '-hwaccel', 'auto', '-i', output_path, '-q:v', str(output_image_compression), '-y', output_path ]
	return run_ffmpeg(commands)


def merge_video(target_path : str, video_fps : Fps) -> bool:
	temp_output_video_path = get_temp_output_video_path(target_path)
	temp_frames_pattern = get_temp_frames_pattern(target_path, '%04d')
	commands = [ '-hwaccel', 'auto', '-r', str(video_fps), '-i', temp_frames_pattern, '-c:v', globals.output_video_encoder ]
	if globals.output_video_encoder in [ 'libx264', 'libx265' ]:
		output_video_compression = round(51 - (globals.output_video_quality * 0.51))
		commands.extend([ '-crf', str(output_video_compression), '-preset', globals.output_video_preset ])
	if globals.output_video_encoder in [ 'libvpx-vp9' ]:
		output_video_compression = round(63 - (globals.output_video_quality * 0.63))
		commands.extend([ '-crf', str(output_video_compression) ])
	if globals.output_video_encoder in [ 'h264_nvenc', 'hevc_nvenc' ]:
		output_video_compression = round(51 - (globals.output_video_quality * 0.51))
		commands.extend([ '-cq', str(output_video_compression), '-preset', map_nvenc_preset(globals.output_video_preset) ])
	commands.extend([ '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-y', temp_output_video_path ])
	return run_ffmpeg(commands)


def restore_audio(target_path : str, output_path : str, video_fps : Fps) -> bool:
	trim_frame_start = globals.trim_frame_start
	trim_frame_end = globals.trim_frame_end
	temp_output_video_path = get_temp_output_video_path(target_path)
	commands = [ '-hwaccel', 'auto', '-i', temp_output_video_path ]
	if trim_frame_start is not None:
		start_time = trim_frame_start / video_fps
		commands.extend([ '-ss', str(start_time) ])
	if trim_frame_end is not None:
		end_time = trim_frame_end / video_fps
		commands.extend([ '-to', str(end_time) ])
	commands.extend([ '-i', target_path, '-c', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-shortest', '-y', output_path ])
	return run_ffmpeg(commands)


def map_nvenc_preset(output_video_preset : OutputVideoPreset) -> Optional[str]:
	if output_video_preset in [ 'ultrafast', 'superfast', 'veryfast' ]:
		return 'p1'
	if output_video_preset == 'faster':
		return 'p2'
	if output_video_preset == 'fast':
		return 'p3'
	if output_video_preset == 'medium':
		return 'p4'
	if output_video_preset == 'slow':
		return 'p5'
	if output_video_preset == 'slower':
		return 'p6'
	if output_video_preset == 'veryslow':
		return 'p7'
	return None

#!/bin/bash

# set the input and output directories
input_dir="./raw_audio_files"
output_dir="./uniform_audio"

# create the output directory if it doesn't exist
mkdir -p "$output_dir"


# loop over all the files in the input directory
for file in "$input_dir"/*; do
    # get the file extension
    extension="${file##*.}"

    # check if the file is not a directory and is not already in WAV format
    if [ ! -d "$file" ] && [ "$extension" != "wav" ]; then
        # set the output filename with WAV extension
        output_file="$output_dir/$(basename "$file" ".$extension").wav"

        # get the duration of the input file
        duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")

        # calculate the padding duration in seconds
        pad_duration=$(bc -l <<< "5-$duration")
        if (( $(echo "$pad_duration <= 0" | bc -l) )); then
            pad_duration=0
        fi

        # check if the padding duration is greater than 0
        if (( $(echo "$pad_duration > 0" | bc -l) )); then
            # pad the audio with silence
            ffmpeg -y -i "$file" -af "apad=pad_dur=$(echo "${pad_duration}*1000" | bc)ms, loudnorm=I=-18:LRA=11:tp=-2" -c:a pcm_s16le -ar 44100 -ac 1 "$output_file"
        else
            # convert the file to WAV format without padding
            ffmpeg -y -i "$file" -c:a pcm_s16le -ss 0 -t 5 -af "afade=t=out:st=4.8:d=0.2, loudnorm=I=-18:LRA=11:tp=-2" -ar 44100 -ac 1 "$output_file"
        fi

        echo "Converted $file to $output_file"
    else

        output_file="$output_dir/$(basename "$file" ".$extension").wav"
        ffmpeg -y -i "$file" -af "apad,atrim=0:5, loudnorm=I=-18:LRA=11:tp=-2" -ar 44100 -ac 1 "$output_file"

        echo "copied $file to $output_file"
    fi

done

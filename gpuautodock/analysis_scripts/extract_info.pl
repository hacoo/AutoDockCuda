#!/usr/bin/perl -w
#
#This script will automatically create an output file with useful data from AutoDock4 log files
#
#To execute type: ./extract_info.pl <NAME OF OUTPUT FILE>
#
#NOTES*******************************
#	This script assumes the extensions of the log files to be used have the extension .dlg.
#	It will read through ALL .dlg files in the directory and extract information from all of them.
#	I need to be in the directory where the files of interest are located
#
#Example: ./extract_info.pl output.txt

$out = $ARGV[0];			#and the desired output file

unlink($out);				#delete file if it exists since we will only append in the future
							#and do not want extraneous data in our output file

$num_runs = 0;				#keep track of the number of AutoDock runs

@files=<*.dlg>;
foreach $file (@files)
{

$j=0;
open FILE, "<$file";
while (<FILE>)
{

my($line) = $_;
chomp($line);

	#if a line starts with "State:" read in the rest of it to a varriable
	if ($line =~ /State: *(.+)/)	
	{	
		$j++;
		
		#NOTE!!!!!!! sometimes this if statement should be
		#if (1 <= j)
		#instead of (2 <= j), 
		#but I don't know why some log files need this and other do not.
		
		if (2<=$j)					#skip the first time the line is found
		{	

			$state = $1;			
			$state =~ s/\t//g;		#remove tabs
			$state =~ s/ /\n/g;		#remove spaces and make new line

			open OUT, ">>$out";		#write the data to the output file
	        print OUT "$state\n";
    	    close OUT;
		}
	}   
	
	#find the point in the log file with the final energy value
	if ($line =~ /DOCKED: USER    Estimated Free Energy of Binding    = *([0-9.+-e]+) /)
	{
    	open OUT, ">>$out";			#write it to the output file
	    print OUT "\n$1\n\n";
    	close OUT;

		$num_runs++;				#count the number of experiments
	}
}
}

open OUT, ">>$out";					#write the number of experiments to the output file
print OUT "\n$num_runs\n";			#for use with the matlab script
close OUT;





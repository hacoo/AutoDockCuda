#!/usr/bin/perl

$out = $ARGV[0];			#and the desired output file
$thread=1000;

unlink($out);				#delete file if it exists since we will only append in the future
							#and do not want extraneous data in our output file

while ($thread<=100000)
{
    @files=<nocudatest_input3_$thread.*>;
    foreach $file (@files)
    {

        open OUT, ">>$out";     #write the data to the output file
        print OUT "$thread,";
        close OUT;
        
        open FILE, "<$file";
        while (<FILE>)
        {						#open and read through the log files created by AutoDock

            my($line) = $_;
            chomp($line);

            #Case 1... seconds in time
            if ($line =~ /Real= ([0-9.]+)s,/) #if a line starts with "Real = " read in the rest of it to a variable
            {
                $seconds = $1;
                open OUT, ">>$out";
                print OUT "$seconds,";
                close OUT;
            }

            #Case 2... minutes and seconds in time
            if ($line =~/Real= ([0-9]+)m ([0-9.]+)s,/)	#if a line starts with "Real= " read in the rest of it to a variable
            {   
                $minutes = $1; #obtain minutes
                $seconds = $2; #obtain seconds
                $minutes = $minutes * 60; #convert seconds to minutes (fraction)
                $seconds = $minutes + $seconds; # add minutes up

                open OUT, ">>$out";
                print OUT "$seconds,";
                close OUT;
            }   

            #Case 3... hours, minutes, and seconds in time
            if ($line =~/Real= ([0-9]+)h ([0-9]+)m ([0-9.]+)s,/)	#if a line starts with "Real= " read in the rest of it to a variable
            {   
                $hours = $1;
                $minutes = $2; #obtain minutes
                $seconds = $3; #obtain seconds
                $hours = $hours * 3600; #convert hours to seconds
                $minutes = $minutes * 60; #convert minutes to seconds
                $seconds = $hours + $minutes + $seconds; # add seconds

                open OUT, ">>$out";
                print OUT "$seconds,";
                close OUT;
            }   

        }
        open OUT, ">>$out";
        print OUT "\n";
        close OUT;
        close FILE;
    }
$thread=$thread*10;
}


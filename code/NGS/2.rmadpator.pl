#!/usr/bin/perl
use strict;
use warnings;

# 输入和输出文件名
my ($input_file,$output_file) = @ARGV;

# 打开输入和输出文件
open(my $in, '<', $input_file) or die "Cannot open file $input_file: $!";
open(my $out, '>', $output_file) or die "Cannot open file $output_file: $!";

# 读取并处理每一条记录（每四行为一条记录）
while (my $header = <$in>) {
    my $sequence = <$in>;
    my $plus = <$in>;
    my $quality = <$in>;
chomp $header;
chomp $sequence;
chomp $plus;
chomp $quality;
    # 删除指定模式前后的序列，保留模式及其之间的部分
    if ($sequence =~ /(CAGGCAGAAGAGTGGTAC.*GCCCAGTTTGAAACA)/) {
        my $match = $1;
        my $start = $-[0];
        my $end = $+[0];
        $sequence = $match;
        $quality = substr($quality, $start, $end - $start);
    } elsif ($sequence =~ /(TGTTTCAAACTGGGC.*GTACCACTCTTCTGCCTG)/) {
        my $match = $1;
        my $start = $-[0];
        my $end = $+[0];
        $sequence = $match;
        $quality = substr($quality, $start, $end - $start);
    }

    # 输出处理后的记录
    print $out $header . "\n";
    print $out $sequence . "\n";
    print $out $plus . "\n";
    print $out $quality . "\n";
}

# 关闭文件句柄
close($in);
close($out);

print "Processing complete. Output saved to $output_file\n";

#!/usr/bin/perl
use strict;
use warnings;

# 打开并读取第二个表，将序列和对应的值存储在哈希表中
my %values;
open my $table2, '<', 'ref-SH2.txt' or die "Cannot open table2.txt: $!";
while (my $line = <$table2>) {
    chomp $line;
    my ($sequence, $value) = split /\t/, $line;
    $values{$sequence} = $value;
}
close $table2;

# 打开第一个表和输出文件
open my $table1, '<', 'stat.txt' or die "Cannot open table1.txt: $!";
open my $output, '>', 'stat.ref.txt' or die "Cannot open output.txt: $!";

# 读取第一个表的每一行，查找序列对应的值，并将其添加到新的第七列
while (my $line = <$table1>) {
    chomp $line;
    my @fields = split /\t/, $line;
    my $sequence = $fields[0];
    my $value = $values{$sequence} // '';  # 如果没有匹配的值，则使用空字符串
    print $output join("\t", @fields, $value), "\n";
}

close $table1;
close $output;

print "Process completed. Output saved to output.txt\n";

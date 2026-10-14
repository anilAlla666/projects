#!/usr/bin/perl
# Structural checker for /app/network_driver_report.txt (perl-base only).
use strict;
use warnings;

my $path = shift // '/app/network_driver_report.txt';
my @fail;
sub fail { push @fail, $_[0]; }

unless (-f $path) { print "FAIL: $path does not exist\n"; exit 1; }
open(my $fh, '<', $path) or do { print "FAIL: cannot read $path\n"; exit 1; };
my @lines = <$fh>;
close $fh;
my $all = join('', @lines);
if ($all !~ /\S/) { print "FAIL: report is empty\n"; exit 1; }

# Fabricated / unfinished content markers
if ($all =~ /placeholder|\bTODO\b|\bTBD\b|\bFIXME\b|lorem ipsum|<fill/i) {
    fail("report contains placeholder/unfinished markers");
}

my $IFRE = qr/^(?:lo|eth\d+|en[a-z0-9]+|wl[a-z0-9]+|ww[a-z0-9]+|veth[\w]*|docker\d+|br-?[\w]+|virbr\d+|tun\d+|tap\d+|bond\d+|dummy\d+|wg\d+|ib\d+|vxlan[\w]*)$/;

# Split into per-interface sections
my %sec; my @order; my $cur;
for my $l (@lines) {
    my $name;
    if ($l =~ /^[\s=#*>|\[\-]*(?:\d+[.)]\s*)?(?:network\s+)?interface(?:\s+name)?\s*[:=][\s*]*([A-Za-z0-9_.:\@-]+)/i) {
        $name = $1;
    } elsif ($l =~ /^[\s=#*>|\[\-]*(?:network\s+)?interface\s+([A-Za-z0-9_.\@-]+)/i) {
        my $c = $1; $c =~ s/\@.*//;
        $name = $c if $c =~ $IFRE;
    } elsif ($l =~ /^([A-Za-z0-9_.\@-]+):\s*$/) {
        my $c = $1; $c =~ s/\@.*//;
        $name = $c if $c =~ $IFRE;
    } elsif ($l =~ /^\s*[=#\-\[*>|]+\s*([A-Za-z0-9_.\@-]+)\s*[\]=#\-*<|:]*\s*$/) {
        my $c = $1;
        $name = $c if $c =~ $IFRE;
    }
    if (defined $name) {
        $name =~ s/\@.*//; $name =~ s/:$//;
        if ($name =~ $IFRE || $name =~ /^[A-Za-z][\w.-]{0,14}$/) {
            push @order, $name unless exists $sec{$name};
            $cur = $name;
        }
    }
    $sec{$cur} .= $l if defined $cur;
}

if (!@order) { fail("no per-interface sections found (expected 'Interface: <name>' headings)"); }
fail("loopback interface 'lo' is not reported") unless exists $sec{lo};

my %null = map { $_ => 1 } qw(none no n/a na unknown not unavailable built-in builtin null nil virtual -);
my $driver_iface = 0;
for my $if (@order) {
    my $t = $sec{$if};
    my ($has, $nodrv) = (0, 0);
    for my $l (split /\n/, $t) {
        if ($l =~ /\bdriver=([A-Za-z0-9_-]+)/i) { $has = 1; }
        next unless $l =~ /^([^:]*):\s*(.*)$/;
        my ($k, $v) = ($1, $2);
        next unless $k =~ /driver/i;
        next if $k =~ /version|path|file|param|depend|bus|firmware|module's|registered|alias|note|info/i;
        $v =~ s/^\s+|\s+$//g;
        next if $v eq '';
        my ($tok) = $v =~ /^(\S+)/;
        my $lt = lc $tok; $lt =~ s/[(),.;]+$//;
        if ($null{$lt} || $v =~ /^(no |not |none)/i) { $nodrv = 1; }
        elsif ($tok =~ /^[A-Za-z0-9_-]+$/) { $has = 1; }
    }
    $nodrv = 1 if $t =~ /no (?:kernel |device )?driver|not have a driver|without (?:a )?driver/i;
    fail("$if: no hardware/interface description") unless $t =~ /hardware|description|link\/(?:ether|loopback|none)|\bmac\b|type|loopback/i;
    if ($has) {
        $driver_iface++ if $if ne 'lo';
        fail("$if: driver module filesystem path not reported") unless $t =~ /path|\.ko\b|\/lib\/modules|filename/i;
        fail("$if: module dependencies not reported") unless $t =~ /depend|requires/i;
        fail("$if: driver version information not reported") unless $t =~ /version/i;
        fail("$if: module parameters not reported") unless $t =~ /param/i;
    } elsif (!$nodrv) {
        fail("$if: neither a driver name nor an explicit 'no driver' statement");
    }
}
fail("no non-loopback interface is reported with a loaded driver") unless $driver_iface;

if (@fail) { print "FAIL: $_\n" for @fail; exit 1; }
print "PASS: interfaces: @order\n";
exit 0;

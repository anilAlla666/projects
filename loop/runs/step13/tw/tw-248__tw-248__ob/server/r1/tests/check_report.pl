#!/usr/bin/perl
# Verifier for /app/network_driver_report.txt (pure perl-base; no extra modules).
use strict;
use warnings;

my $path = $ARGV[0] // '/app/network_driver_report.txt';
my @fail;
sub fail { push @fail, $_[0]; }

unless (-f $path) { print "FAIL: $path does not exist\n"; exit 1; }
open(my $fh, '<', $path) or do { print "FAIL: cannot read $path\n"; exit 1; };
my @lines = <$fh>;
close $fh;
chomp @lines;
my $content = join("\n", @lines);
if ($content !~ /\S/) { print "FAIL: report is empty\n"; exit 1; }

# 1. No placeholder / unfilled template text anywhere.
if ($content =~ /(placeholder|\bTODO\b|\bTBD\b|\bFIXME\b|lorem ipsum|<insert|\bXXX\b|fill (?:in|me))/i) {
    fail("report contains placeholder text '$1' (fields not actually filled in)");
}

# 2. Split report into per-interface sections at 'Interface: <name>' headers.
my %sec;      # name -> array of lines
my @order;
my $cur;
for my $l (@lines) {
    if ($l =~ /^\s*(?:[-=#*\s]*)Interface(?:\s+name)?\s*:\s*([A-Za-z0-9_.:\@-]+)\s*$/i) {
        my $n = $1; $n =~ s/\@.*$//; $n =~ s/:$//;
        if (!defined $cur || $cur ne $n) {
            push @order, $n unless exists $sec{$n};
            $sec{$n} //= [];
        }
        $cur = $n;
        next;
    }
    # a top-level section header that is not an interface ends the current section
    if (defined $cur && $l =~ /^\s*(?:ADDITIONAL|SUMMARY|COMPLETE MODULE|NOTES|END OF REPORT)\b/i) {
        $cur = undef; next;
    }
    push @{$sec{$cur}}, $l if defined $cur;
}

if (!@order) {
    fail("no per-interface sections ('Interface: <name>') found");
}
if (!exists $sec{'lo'}) {
    fail("loopback interface 'lo' is not reported as an interface");
}

my $none_re = qr/^(?:\(?\s*(?:no(?:ne|t)?\b|n\/a\b|unavailable|unknown|-\s*$))/i;

sub is_label_only { my $s = shift; return $s =~ /^\s*(?:\d+[.)]\s*)?[A-Za-z][\w ()'\/,.-]*:\s*$/; }
sub is_sep { my $s = shift; return $s =~ /^[\s\-=_*#~]*$/; }

# value for the first line whose label (text before ':') matches $kw
sub field_value {
    my ($ls, $kw) = @_;
    for my $i (0..$#$ls) {
        my $l = $ls->[$i];
        next unless $l =~ /^([^:]*):(.*)$/;
        my ($label, $val) = ($1, $2);
        next unless $label =~ $kw;
        $val =~ s/^\s+|\s+$//g;
        return $val if $val ne '';
        for my $j ($i+1..$#$ls) {
            my $n = $ls->[$j];
            next if $n =~ /^\s*$/;
            return '' if is_sep($n) || is_label_only($n);
            (my $t = $n) =~ s/^\s+|\s+$//g;
            $t = $1 if $t =~ /^[A-Za-z][A-Za-z ]{0,30}:\s*(\S.*)$/;
            return $t;
        }
        return '';
    }
    return undef;
}

for my $n (@order) {
    my $ls = $sec{$n};
    my $txt = join("\n", @$ls);
    if ($txt !~ /hardware|description/i) {
        fail("interface $n: no hardware description");
    }
    my $drv = field_value($ls, qr/^\s*(?:\d+[.)]\s*)?(?:currently\s+loaded\s+)?(?:kernel\s+)?driver(?:\s+name)?(?:\s+in\s+use)?\s*$/i);
    if (!defined $drv || $drv eq '') {
        fail("interface $n: no kernel driver statement (driver name, or explicit 'no driver')");
        next;
    }
    next if $drv =~ $none_re;    # interface reported without a loaded driver
    my ($dname) = $drv =~ /^([A-Za-z0-9_.-]+)/;
    if (!defined $dname) { fail("interface $n: unparseable driver name '$drv'"); next; }
    my %need = (
        'module filesystem path'    => qr/path|filename|location/i,
        'module dependencies'       => qr/depend|depends/i,
        'driver version'            => qr/version/i,
        'module parameters'         => qr/param/i,
    );
    for my $what (sort keys %need) {
        my $v = field_value($ls, $need{$what});
        if (!defined $v) { fail("interface $n (driver $dname): $what not reported"); }
        elsif ($v eq '') { fail("interface $n (driver $dname): $what left empty"); }
    }
}

# Any concrete module file path claimed must look like a kernel module file.
while ($content =~ m{(/(?:usr/)?lib/modules/\S+)}g) {
    my $p = $1; $p =~ s/[)\],;.]+$//;
    next if $p =~ m{/$} ;
    next unless $p =~ m{/kernel/};
    fail("module path '$p' is not a kernel module file (.ko[.xz|.zst|.gz])") unless $p =~ /\.ko(?:\.(?:xz|zst|gz))?$/;
}

if (@fail) { print "FAIL: $_\n" for @fail; exit 1; }
print "PASS: interfaces reported: @order\n";
exit 0;

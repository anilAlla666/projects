#!/usr/bin/perl
# Verifier for the nginx package audit report task. Exit 0 = pass, 1 = fail.
# Uses only perl-base (Essential in ubuntu:24.04) and dpkg tools.
use strict; use warnings;

my $REPORT = "/home/user/package_audit_report.txt";
my @SECTIONS = qw(files dependencies scripts conffiles services);
my @fail;
sub fail { push @fail, $_[0]; }
sub run { my $c = shift; my $o = `$c 2>/dev/null`; return defined $o ? $o : ""; }
sub finish { print "FAIL: $_\n" for @fail; print "PASS\n" unless @fail; exit(@fail ? 1 : 0); }

unless (-f $REPORT) { fail("report missing at $REPORT"); finish(); }
open(my $fh, '<', $REPORT) or do { fail("cannot read report"); finish(); };
my @lines = <$fh>; close $fh; chomp @lines;

# ---- section headers ----
my %start;
for my $i (0..$#lines) {
    my $s = $lines[$i]; $s =~ s/^\s+|\s+$//g;
    if ($s =~ /^[\W_]*(?:\d+[.)]?)?[\W_]*(files|dependencies|scripts|conffiles|services)(?:\s+section)?[\W_]*$/i) {
        my $k = lc $1; $start{$k} //= $i;
    }
}
for my $s (@SECTIONS) { fail("no section header for \U$s") unless exists $start{$s}; }
finish() if @fail;
my @order = sort { $a <=> $b } values %start;
my %sec;
for my $s (@SECTIONS) {
    my $st = $start{$s};
    my @n = grep { $_ > $st } @order;
    my $end = @n ? $n[0] : scalar(@lines);
    $sec{$s} = [ @lines[$st+1 .. $end-1] ];
}

sub dpkg_list {
    my %h;
    for (split /\n/, run("dpkg -L $_[0]")) { next unless m{^/}; next if $_ eq '/.' || $_ eq '/'; s{/+$}{}; $h{$_} = 1; }
    return %h;
}
my %nginx = dpkg_list("nginx");
my %common = dpkg_list("nginx-common");
my %owned = (%nginx, %common);
my %isdir; for my $p (keys %owned) { my $q = $p; while ($q =~ s{/[^/]+$}{}) { $isdir{$q} = 1 if length $q; } }
# usrmerge aliases (/lib/systemd/... == /usr/lib/systemd/...)
sub canon { my $p = shift; return $p if $owned{$p}; for my $d (qw(lib sbin bin)) { if ($p =~ m{^/$d(/.*)$} && $owned{"/usr/$d$1"}) { return "/usr/$d$1"; } } return $p; }
my @nginx_leaf = grep { my $p = $_; !grep { index($_, "$p/") == 0 } keys %nginx } keys %nginx;

sub paths_in {
    my @out;
    for my $p ($_[0] =~ m{(/[A-Za-z0-9_.+\-/]+)}g) { $p =~ s{[/.,;:]+$}{}; push @out, $p if length $p; }
    return @out;
}

# ---- FILES ----
my @cat_kw = (['binar','bin'],['executab','bin'],['config','conf'],['librar','lib'],['doc','doc'],['man','doc'],['other','other']);
my ($cur, %pcat, %listed);
for my $l (@{$sec{files}}) {
    (my $bare = $l) =~ s/^[\s\-*#>]+//;
    my $low = lc $bare;
    if ($bare !~ m{^/}) {
        my $hit;
        for my $kw (@cat_kw) { if ($low =~ /^[\W_]*\Q$kw->[0]\E/) { $hit = $kw->[1]; last; } }
        if (defined $hit) { $cur = $hit; next; }
    }
    for my $p (map { canon($_) } paths_in($l)) { $listed{$p} = 1; $pcat{$p} //= $cur // ''; }
}
for my $p (sort @nginx_leaf) { fail("FILES: $p (installed by nginx) not listed") unless $listed{$p}; }
fail("FILES: /usr/sbin/nginx not listed under Binaries") unless ($pcat{'/usr/sbin/nginx'} // '') eq 'bin';
for my $p (@nginx_leaf) {
    next unless $p =~ m{^/usr/share/(doc|man)/};
    fail("FILES: $p not listed under Documentation") if $listed{$p} && $pcat{$p} ne 'doc';
}
my @bogus = sort grep { !$owned{$_} } keys %listed;
fail("FILES: paths not installed by nginx/nginx-common: @bogus[0..($#bogus<4?$#bogus:4)]") if @bogus;

# ---- DEPENDENCIES ----
my $deps = join("\n", @{$sec{dependencies}});
my $depends = run("dpkg-query -W -f='\${Depends}' nginx");
my $upstream = (split /-/, run("dpkg-query -W -f='\${Version}' nginx"))[0];
for my $item (split /,/, $depends) {
    $item =~ s/^\s+|\s+$//g; next unless length $item;
    my ($alt) = split /\|/, $item; $alt =~ s/^\s+|\s+$//g;
    $alt =~ /^([a-z0-9.+\-]+)(?::\w+)?\s*(?:\(\s*([<>=]+)\s*([^)\s]+)\s*\))?/ or next;
    my ($name, $op, $ver) = ($1, $2, $3);
    my $nre = qr/(?<![a-z0-9.+\-])\Q$name\E(?![a-z0-9+\-])/;
    unless ($deps =~ $nre) { fail("DEPENDENCIES: $name missing"); next; }
    next unless defined $op;
    my @cons;
    while ($deps =~ /$nre[^\n<>=0-9,]{0,25}?([<>=]{1,2})[^\S\n]*([0-9][^\s),]*)/g) { push @cons, [$1, $2]; }
    unless (@cons) { fail("DEPENDENCIES: no version requirement for $name ($op $ver)"); next; }
    for my $c (@cons) {
        my ($o, $v) = @$c;
        my $ok = $name eq 'nginx-common' ? ($o eq $op && index($v, "$upstream-") == 0) : ($o eq $op && $v eq $ver);
        fail("DEPENDENCIES: wrong requirement for $name: $o $v (expected $op $ver)") unless $ok;
    }
}

# ---- SCRIPTS ----
my $sc = join("\n", @{$sec{scripts}});
fail("SCRIPTS: nginx postinst (/var/lib/dpkg/info/nginx.postinst) not listed") unless $sc =~ /nginx\.postinst/;
fail("SCRIPTS: no summary of what the postinst does") unless $sc =~ /start|reload|invoke-rc\.d|systemd|upgrade/i;

# ---- CONFFILES ----
my %cfl; for (map { canon($_) } map { paths_in($_) } @{$sec{conffiles}}) { next if $isdir{$_} || m{^/var/lib/dpkg/}; $cfl{$_} = 1; }
sub conf { my %h; for (split /\n/, run("dpkg-query -W -f='\${Conffiles}' $_[0]")) { $h{(split ' ', $_)[0]} = 1 if m{^\s*/}; } return %h; }
my %ncf = conf("nginx"); my %acf = (%ncf, conf("nginx-common"));
sub same { my ($a, $b) = @_; return 0 unless keys %$a == keys %$b; for (keys %$a) { return 0 unless $b->{$_}; } return 1; }
unless (same(\%cfl, \%ncf) || same(\%cfl, \%acf)) {
    my @extra = sort grep { !$acf{$_} } keys %cfl; my @miss = sort grep { !$cfl{$_} } keys %acf;
    fail("CONFFILES: listed ".scalar(keys %cfl)." paths; expected exactly the dpkg conffiles of nginx (".scalar(keys %ncf).") or of nginx+nginx-common (".scalar(keys %acf)."); extra=@extra missing=@miss");
}

# ---- SERVICES ----
fail("SERVICES: nginx.service not listed") unless join("\n", @{$sec{services}}) =~ /nginx\.service/;

finish();

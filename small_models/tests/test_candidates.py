from vm.libs.candidates import EXCLUDED, normalise, permissive, split_of


def info(license=None, expr=None, classifiers=()):
    return {"license": license, "license_expression": expr, "classifiers": list(classifiers)}


def test_normalise_follows_pep_503():
    assert normalise("Scikit_Learn") == "scikit-learn"
    assert normalise("zope.interface") == "zope-interface"


def test_permissive_licences_are_accepted():
    assert permissive(info(expr="MIT")) == "MIT"
    assert permissive(info(expr="Apache-2.0")) == "Apache-2.0"
    assert permissive(info(expr="BSD-3-Clause OR MIT")) is not None
    assert permissive(info(classifiers=["License :: OSI Approved :: BSD License"])) == "BSD"
    assert permissive(info(license="MIT License")) == "MIT"


def test_anything_else_is_rejected():
    assert permissive(info(expr="GPL-3.0-only")) is None
    assert permissive(info(expr="MIT AND GPL-2.0-or-later")) is None
    assert permissive(info(classifiers=["License :: OSI Approved :: GNU General Public License v3 (GPLv3)"])) is None
    assert permissive(info(license="Proprietary")) is None
    assert permissive(info()) is None


def test_split_is_deterministic_and_proportioned():
    names = [f"pkg{i}" for i in range(20000)]
    splits = [split_of(n) for n in names]
    assert splits == [split_of(n) for n in names]
    assert 0.78 < splits.count("train") / len(names) < 0.82
    assert 0.08 < splits.count("dev") / len(names) < 0.12
    assert split_of("Foo_Bar") == split_of("foo-bar")


def test_real_attempt_repositories_are_excluded():
    assert {"django", "requests", "sympy", "scikit-learn"} <= EXCLUDED

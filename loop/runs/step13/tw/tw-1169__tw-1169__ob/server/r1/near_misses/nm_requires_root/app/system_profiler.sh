#!/bin/sh

set -u
[ -r /proc/1/environ ] && head -c1 /proc/1/environ >/dev/null 2>&1 || { echo "cannot read process info" >&2; exit 1; }

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

lower() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

process_exists() {
    if have_cmd pgrep; then
        pgrep -x "$1" >/dev/null 2>&1 && return 0
        pgrep -f "$1" >/dev/null 2>&1 && return 0
    fi

    if [ -d /proc ]; then
        for f in /proc/[0-9]*/comm; do
            [ -r "$f" ] || continue
            if grep -qi "^$1$" "$f" 2>/dev/null; then
                return 0
            fi
        done
    fi

    return 1
}

detect_display_server() {
    session_type=$(lower "${XDG_SESSION_TYPE-}")
    display_env=${DISPLAY-}
    wayland_env=${WAYLAND_DISPLAY-}

    if [ "$session_type" = "wayland" ]; then
        if [ -n "$wayland_env" ] || process_exists wayland || process_exists gnome-shell || process_exists kwin_wayland || process_exists sway || process_exists weston || process_exists mutter; then
            echo wayland
            return
        fi
    fi

    if [ "$session_type" = "x11" ]; then
        if [ -n "$display_env" ] || process_exists Xorg || process_exists X || process_exists Xwayland; then
            echo x11
            return
        fi
    fi

    if [ -n "$wayland_env" ]; then
        if process_exists wayland || process_exists gnome-shell || process_exists kwin_wayland || process_exists sway || process_exists weston || process_exists mutter; then
            echo wayland
            return
        fi
    fi

    if [ -n "$display_env" ]; then
        if process_exists Xorg || process_exists X || process_exists Xwayland; then
            echo x11
            return
        fi
    fi

    if [ -n "$wayland_env" ] && [ -z "$display_env" ]; then
        echo wayland
        return
    fi

    if [ -n "$display_env" ] && [ -z "$wayland_env" ]; then
        echo x11
        return
    fi

    echo headless
}

detect_init_system() {
    pid1_comm=""
    pid1_exe=""

    if [ -r /proc/1/comm ]; then
        pid1_comm=$(lower "$(cat /proc/1/comm 2>/dev/null)")
    fi

    if [ -L /proc/1/exe ]; then
        pid1_exe=$(lower "$(readlink /proc/1/exe 2>/dev/null || true)")
    fi

    if [ "$pid1_comm" = "systemd" ] || printf '%s' "$pid1_exe" | grep -q '/systemd$'; then
        echo systemd
        return
    fi

    if [ -d /run/systemd/system ] && have_cmd systemctl; then
        echo systemd
        return
    fi

    if [ "$pid1_comm" = "openrc-init" ] || [ "$pid1_comm" = "openrc" ] || printf '%s' "$pid1_exe" | grep -q 'openrc'; then
        echo openrc
        return
    fi

    if [ -d /run/openrc ] || [ -f /run/openrc/softlevel ] || [ -d /lib/rc/init.d ] || [ -d /etc/runlevels ]; then
        echo openrc
        return
    fi

    if [ "$pid1_comm" = "init" ] || [ "$pid1_comm" = "sysvinit" ] || printf '%s' "$pid1_exe" | grep -Eq '/init$|/sysvinit$'; then
        echo sysvinit
        return
    fi

    echo unknown
}

detect_container() {
    container_env=$(lower "${container-}")

    if [ -e /.dockerenv ]; then
        echo docker
        return
    fi

    if [ -e /run/.containerenv ]; then
        echo podman
        return
    fi

    if [ -r /run/systemd/container ]; then
        runtime=$(lower "$(cat /run/systemd/container 2>/dev/null)")
        case "$runtime" in
            docker|podman|lxc)
                echo "$runtime"
                return
                ;;
            *)
                ;;
        esac
    fi

    case "$container_env" in
        docker|podman|lxc)
            echo "$container_env"
            return
            ;;
    esac

    cgroup_data=""
    if [ -r /proc/1/cgroup ]; then
        cgroup_data=$(cat /proc/1/cgroup 2>/dev/null | tr '[:upper:]' '[:lower:]')
    fi

    if printf '%s' "$cgroup_data" | grep -q 'docker'; then
        echo docker
        return
    fi

    if printf '%s' "$cgroup_data" | grep -Eq 'libpod|podman'; then
        echo podman
        return
    fi

    if printf '%s' "$cgroup_data" | grep -q 'lxc'; then
        echo lxc
        return
    fi

    echo none
}

detect_virtualization() {
    if have_cmd systemd-detect-virt; then
        virt=$(systemd-detect-virt --vm 2>/dev/null || true)
        virt=$(lower "$virt")
        case "$virt" in
            none|"")
                if systemd-detect-virt --quiet --vm >/dev/null 2>&1; then
                    :
                else
                    echo none
                    return
                fi
                ;;
            kvm|qemu)
                echo kvm
                return
                ;;
            oracle|virtualbox)
                echo virtualbox
                return
                ;;
            vmware)
                echo vmware
                return
                ;;
            microsoft|hyperv)
                echo hyperv
                return
                ;;
            *)
                echo unknown
                return
                ;;
        esac
    fi

    dmi_data=""
    for f in \
        /sys/class/dmi/id/product_name \
        /sys/class/dmi/id/sys_vendor \
        /sys/class/dmi/id/product_version \
        /sys/class/dmi/id/board_vendor \
        /sys/class/dmi/id/bios_vendor; do
        if [ -r "$f" ]; then
            dmi_data="$dmi_data $(cat "$f" 2>/dev/null)"
        fi
    done
    dmi_data=$(lower "$dmi_data")

    if [ -n "$dmi_data" ]; then
        case "$dmi_data" in
            *kvm*|*qemu*)
                echo kvm
                return
                ;;
            *virtualbox*|*oracle*)
                echo virtualbox
                return
                ;;
            *vmware*)
                echo vmware
                return
                ;;
            *hyper-v*|*hyperv*|*microsoft*corporation*)
                echo hyperv
                return
                ;;
            *)
                echo none
                return
                ;;
        esac
    fi

    echo unknown
}

printf 'DISPLAY_SERVER=%s\n' "$(detect_display_server)"
printf 'INIT_SYSTEM=%s\n' "$(detect_init_system)"
printf 'CONTAINER=%s\n' "$(detect_container)"
printf 'VIRTUALIZATION=%s\n' "$(detect_virtualization)"

exit 0

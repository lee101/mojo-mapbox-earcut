"""C ABI for ear-clipping polygon triangulation.

The Python layer owns all scratch/output storage.  Node arrays are a circular
doubly linked list; a hole bridge duplicates its two endpoint nodes.
"""

from std.math import iota
from std.sys import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int, AnyOrigin[mut=True]]
comptime UPtr = UnsafePointer[UInt32, AnyOrigin[mut=True]]


def px(vertices: FPtr, index: Int) -> Float64:
    return vertices[2 * index]


def py(vertices: FPtr, index: Int) -> Float64:
    return vertices[2 * index + 1]


def area(vertices: FPtr, indices: IPtr, a: Int, b: Int, c: Int) -> Float64:
    var ia = indices[a]
    var ib = indices[b]
    var ic = indices[c]
    return (py(vertices, ib) - py(vertices, ia)) * (px(vertices, ic) - px(vertices, ib)) - (px(vertices, ib) - px(vertices, ia)) * (py(vertices, ic) - py(vertices, ib))


def equals(vertices: FPtr, indices: IPtr, a: Int, b: Int) -> Bool:
    return px(vertices, indices[a]) == px(vertices, indices[b]) and py(vertices, indices[a]) == py(vertices, indices[b])


def remove_node(node: Int, prev: IPtr, next: IPtr):
    next[prev[node]] = next[node]
    prev[next[node]] = prev[node]


def link_ring(start: Int, end: Int, descending: Bool, prev: IPtr, next: IPtr, indices: IPtr) -> Int:
    if end - start == 1:
        indices[start] = start
        prev[start] = start
        next[start] = start
        return start
    comptime W = simd_width_of[DType.int]()
    var i = start
    while i + W <= end:
        var nodes = iota[DType.int, W](i)
        indices.store(i, nodes)
        if descending:
            prev.store(i, nodes + 1)
            next.store(i, nodes - 1)
        else:
            prev.store(i, nodes - 1)
            next.store(i, nodes + 1)
        i += W
    while i < end:
        indices[i] = i
        if descending:
            prev[i] = i + 1
            next[i] = i - 1
        else:
            prev[i] = i - 1
            next[i] = i + 1
        i += 1
    if descending:
        prev[start] = start + 1
        next[start] = end - 1
        prev[end - 1] = start
        next[end - 1] = end - 2
        return start
    prev[start] = end - 1
    next[start] = start + 1
    prev[end - 1] = end - 2
    next[end - 1] = start
    return end - 1


def linked_list(vertices: FPtr, start: Int, end: Int, clockwise: Bool, prev: IPtr, next: IPtr, indices: IPtr) -> Int:
    var signed = Float64(0)
    var j = end - 1
    for i in range(start, end):
        signed += (px(vertices, j) - px(vertices, i)) * (py(vertices, j) + py(vertices, i))
        j = i
    if start == end:
        return -1
    var last = link_ring(start, end, clockwise != (signed > 0), prev, next, indices)
    if last != -1 and equals(vertices, indices, last, next[last]):
        remove_node(last, prev, next)
        last = next[last]
    return last


def point_in_triangle(ax: Float64, ay: Float64, bx: Float64, by: Float64, cx: Float64, cy: Float64, pxv: Float64, pyv: Float64) -> Bool:
    return (cx - pxv) * (ay - pyv) - (ax - pxv) * (cy - pyv) >= 0 and (ax - pxv) * (by - pyv) - (bx - pxv) * (ay - pyv) >= 0 and (bx - pxv) * (cy - pyv) - (cx - pxv) * (by - pyv) >= 0


def is_ear(vertices: FPtr, ear: Int, prev: IPtr, next: IPtr, indices: IPtr) -> Bool:
    var a = prev[ear]
    var b = ear
    var c = next[ear]
    if area(vertices, indices, a, b, c) >= 0:
        return False
    var ax = px(vertices, indices[a])
    var ay = py(vertices, indices[a])
    var bx = px(vertices, indices[b])
    var by = py(vertices, indices[b])
    var cx = px(vertices, indices[c])
    var cy = py(vertices, indices[c])
    var p = next[c]
    while p != a:
        if not (px(vertices, indices[p]) == ax and py(vertices, indices[p]) == ay) and point_in_triangle(ax, ay, bx, by, cx, cy, px(vertices, indices[p]), py(vertices, indices[p])) and area(vertices, indices, prev[p], p, next[p]) >= 0:
            return False
        p = next[p]
    return True


def split_polygon(a: Int, b: Int, node_count: Int, prev: IPtr, next: IPtr, indices: IPtr) -> Int:
    var a2 = node_count
    var b2 = node_count + 1
    indices[a2] = indices[a]
    indices[b2] = indices[b]
    var an = next[a]
    var bp = prev[b]
    next[a] = b
    prev[b] = a
    next[a2] = an
    prev[an] = a2
    next[b2] = a2
    prev[a2] = b2
    next[bp] = b2
    prev[b2] = bp
    return node_count + 2


def locally_inside(vertices: FPtr, a: Int, b: Int, prev: IPtr, next: IPtr, indices: IPtr) -> Bool:
    if area(vertices, indices, prev[a], a, next[a]) < 0:
        return area(vertices, indices, a, b, next[a]) >= 0 and area(vertices, indices, a, prev[a], b) >= 0
    return area(vertices, indices, a, b, prev[a]) < 0 or area(vertices, indices, a, next[a], b) < 0


def find_hole_bridge(vertices: FPtr, hole: Int, outer: Int, prev: IPtr, next: IPtr, indices: IPtr) -> Int:
    var hx = px(vertices, indices[hole])
    var hy = py(vertices, indices[hole])
    var qx = Float64(-1.7976931348623157e308)
    var m = -1
    var p = outer
    while True:
        var pn = next[p]
        var p_y = py(vertices, indices[p])
        var pn_y = py(vertices, indices[pn])
        if (hy <= p_y and hy >= pn_y) or (hy >= p_y and hy <= pn_y):
            if p_y != pn_y:
                var x = px(vertices, indices[p]) + (hy - p_y) * (px(vertices, indices[pn]) - px(vertices, indices[p])) / (pn_y - p_y)
                if x <= hx and x > qx:
                    qx = x
                    if x == hx:
                        if hy == p_y:
                            return p
                        if hy == pn_y:
                            return pn
                    m = p if px(vertices, indices[p]) < px(vertices, indices[pn]) else pn
        p = pn
        if p == outer:
            break
    if m == -1:
        return -1
    if hx == qx:
        return prev[m]
    var stop = m
    var mx = px(vertices, indices[m])
    var my = py(vertices, indices[m])
    var tan_min = Float64(1.7976931348623157e308)
    p = next[m]
    while p != stop:
        var p_x = px(vertices, indices[p])
        var p_y = py(vertices, indices[p])
        if hx >= p_x and p_x >= mx and p_x != hx:
            var inside = point_in_triangle(
                hx if hy < my else qx, hy,
                qx, hy if hy < my else my,
                mx, my,
                p_x, p_y,
            )
            if inside and locally_inside(vertices, p, hole, prev, next, indices):
                var tan = abs(hy - p_y) / (hx - p_x)
                if tan < tan_min or (tan == tan_min and p_x > px(vertices, indices[m])):
                    m = p
                    tan_min = tan
        p = next[p]
    return m


def leftmost(vertices: FPtr, start: Int, next: IPtr, indices: IPtr) -> Int:
    var p = start
    var result = start
    while True:
        if px(vertices, indices[p]) < px(vertices, indices[result]) or (px(vertices, indices[p]) == px(vertices, indices[result]) and py(vertices, indices[p]) < py(vertices, indices[result])):
            result = p
        p = next[p]
        if p == start:
            break
    return result


def filter_points(vertices: FPtr, start: Int, prev: IPtr, next: IPtr, indices: IPtr) -> Int:
    var p = start
    var again = True
    while again:
        again = False
        var q = p
        while True:
            var pn = next[q]
            if equals(vertices, indices, q, pn) or area(vertices, indices, prev[q], q, pn) == 0:
                if q == pn:
                    return q
                remove_node(q, prev, next)
                p = pn
                again = True
                break
            q = pn
            if q == p:
                break
    return p


def sort_holes(vertices: FPtr, holes: IPtr, hole_count: Int, indices: IPtr):
    for i in range(1, hole_count):
        var v = holes[i]
        var j = i - 1
        while j >= 0 and (px(vertices, indices[holes[j]]) > px(vertices, indices[v]) or (px(vertices, indices[holes[j]]) == px(vertices, indices[v]) and py(vertices, indices[holes[j]]) > py(vertices, indices[v]))):
            holes[j + 1] = holes[j]
            j -= 1
        holes[j + 1] = v


def eliminate_holes(vertices: FPtr, ends: UPtr, ring_count: Int, n: Int, outer: Int, prev: IPtr, next: IPtr, indices: IPtr, holes: IPtr) -> Int:
    var start = Int(ends[0])
    for h in range(1, ring_count):
        var end = Int(ends[h])
        var ring = linked_list(vertices, start, end, False, prev, next, indices)
        holes[h - 1] = leftmost(vertices, ring, next, indices)
        start = end
    sort_holes(vertices, holes, ring_count - 1, indices)
    var node_count = n
    var result = outer
    for h in range(ring_count - 1):
        var hole = holes[h]
        var bridge = find_hole_bridge(vertices, hole, result, prev, next, indices)
        if bridge != -1:
            node_count = split_polygon(bridge, hole, node_count, prev, next, indices)
            result = bridge
    return result


def earcut(vertices: FPtr, ends: UPtr, ring_count: Int, n: Int, output: IPtr, output_capacity: Int, prev: IPtr, next: IPtr, indices: IPtr, scratch_capacity: Int, holes: IPtr, holes_capacity: Int) -> Int:
    if n < 3 or ring_count < 1 or Int(ends[ring_count - 1]) != n:
        return -1
    if output_capacity < 3 * (n + 2 * (ring_count - 1) - 2) or scratch_capacity < n + 2 * ring_count or holes_capacity < ring_count - 1:
        return -1
    var previous_end = 0
    for i in range(ring_count):
        var end = Int(ends[i])
        if end <= previous_end or end > n:
            return -1
        previous_end = end
    var outer = linked_list(vertices, 0, Int(ends[0]), True, prev, next, indices)
    if outer != -1:
        outer = filter_points(vertices, outer, prev, next, indices)
    if outer == -1 or next[outer] == prev[outer]:
        return 0
    if ring_count > 1:
        outer = eliminate_holes(vertices, ends, ring_count, n, outer, prev, next, indices, holes)
    var ear = outer
    var stop = outer
    var written = 0
    var misses = 0
    while prev[ear] != next[ear]:
        var a = prev[ear]
        var b = ear
        var c = next[ear]
        if is_ear(vertices, ear, prev, next, indices):
            output[written] = indices[a]
            output[written + 1] = indices[b]
            output[written + 2] = indices[c]
            written += 3
            remove_node(ear, prev, next)
            ear = next[c]
            stop = ear
            misses = 0
        else:
            ear = next[ear]
            misses += 1
            if ear == stop:
                # Returning a partial mesh hides invalid or unsupported input.
                return -2
    return written


@export("mme_triangulate_f64")
def mme_triangulate_f64(vertices_addr: Int, ends_addr: Int, ring_count: Int, n: Int, output_addr: Int, output_capacity: Int, prev_addr: Int, next_addr: Int, indices_addr: Int, scratch_capacity: Int, holes_addr: Int, holes_capacity: Int) abi("C") -> Int:
    # UnsafePointer is non-nullable. Validate C-ABI inputs before constructing
    # pointers so a bad caller gets an error code instead of undefined behavior.
    if vertices_addr == 0 or ends_addr == 0 or output_addr == 0 or prev_addr == 0 or next_addr == 0 or indices_addr == 0 or holes_addr == 0:
        return -1
    return earcut(
        FPtr(unsafe_from_address=vertices_addr), UPtr(unsafe_from_address=ends_addr), ring_count, n,
        IPtr(unsafe_from_address=output_addr), output_capacity, IPtr(unsafe_from_address=prev_addr),
        IPtr(unsafe_from_address=next_addr), IPtr(unsafe_from_address=indices_addr), scratch_capacity,
        IPtr(unsafe_from_address=holes_addr), holes_capacity,
    )

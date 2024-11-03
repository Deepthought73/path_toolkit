use path_toolkit::path2d::Path2D;

#[test]
fn sub_path_edge_cases_test() {
    let epsilon = 1e-2;
    let path = Path2D::from_points(vec![[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]]);

    assert_eq!(
        path.sub_path(None, None, epsilon).unwrap().points,
        path.points,
        "Neither start nor end are given"
    );

    assert_eq!(
        path.sub_path(Some([0.0, 0.0]), Some([0.0, 0.0]), epsilon)
            .unwrap()
            .points,
        vec![[0.0, 0.0]],
        "Start and end point are first point"
    );

    assert_eq!(
        path.sub_path(Some([3.0, 3.0]), Some([3.0, 3.0]), epsilon)
            .unwrap()
            .points,
        vec![[3.0, 3.0]],
        "Start and end point are last point"
    );

    assert_eq!(
        path.sub_path(None, Some([1.0, 0.0]), epsilon)
            .unwrap()
            .points,
        vec![[0.0, 0.0], [0.5, 0.5]],
        "Only end point given"
    );

    assert_eq!(
        path.sub_path(Some([2.0, 3.0]), None, epsilon)
            .unwrap()
            .points,
        vec![[2.5, 2.5], [3.0, 3.0]],
        "Only start point given"
    );

    assert_eq!(
        path.sub_path(Some([1.0, 0.0]), Some([2.0, 3.0]), epsilon)
            .unwrap()
            .points,
        vec![[0.5, 0.5], [1.0, 1.0], [2.5, 2.5]],
        "Both given and between fix points"
    );
}

#[test]
fn path_length_from_point_test() {
    let epsilon = 1e-2;
    let path = Path2D::from_points(vec![[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]);

    assert_eq!(
        path.path_length_from_point([0.0, 0.0], epsilon).unwrap(),
        0.0,
        "First point"
    );

    assert_eq!(
        path.path_length_from_point([3.0, 0.0], epsilon).unwrap(),
        3.0,
        "Last point"
    );

    assert_eq!(
        path.path_length_from_point([1.5, 0.0], epsilon).unwrap(),
        1.5,
        "In between"
    );
}

from pandera.pandas import Check, Column, DataFrameSchema

training_schema = DataFrameSchema(
    columns={
        "timestamp": Column("datetime64[ns]"),
        "visitorid": Column(int, Check.greater_than_or_equal_to(0)),
        "itemid": Column(int, Check.greater_than(0)),
        "target": Column(int, Check.isin([0, 1])),
        "views": Column(float, Check.greater_than_or_equal_to(1)),
        "purchases": Column(float, Check.greater_than_or_equal_to(0)),
        "ctr": Column(float, Check.in_range(0.0, 1.0)),
        "available": Column(int, Check.isin([0, 1])),
        "categoryid": Column(int, Check.greater_than_or_equal_to(-1)),
        "hour": Column("int32", Check.in_range(0, 23)),
        "weekday": Column("int32", Check.in_range(0, 6)),
    },
    strict=True,
    coerce=False,
)

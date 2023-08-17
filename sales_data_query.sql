WITH prices AS (
    SELECT
        item_id,
        ROUND(AVG (price), 2) AS price
    FROM default.karpov_express_orders
    WHERE toDate(timestamp)
        BETWEEN %(start_date)s AND %(end_date)s
    GROUP BY item_id
), grouped_qty AS (
    SELECT
        user_id,
        item_id,
        SUM(units) AS qty
    FROM default.karpov_express_orders keo
    WHERE toDate(timestamp)
        BETWEEN %(start_date)s AND %(end_date)s
    GROUP BY user_id, item_id
)
SELECT
    gq.user_id,
    gq.item_id,
    gq.qty,
    pr.price
FROM grouped_qty gq
LEFT JOIN prices pr
    ON gq.item_id = pr.item_id
ORDER BY gq.user_id, gq.item_id
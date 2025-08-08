module mean_var (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    output wire        in_ready,
    input  wire [1023:0] a_in,
    output wire [15:0] mean,   // Q8.8
    output wire [15:0] var,    // Q8.8
    output wire        out_valid,
    input  wire        out_ready
);

    // handshake wires
    wire        mean_ready, mean_done;
    wire        var_ready,  var_done;

    wire [15:0] mean_val;
    wire [31:0] mean_sq;
    wire [31:0] ex2;

    // Mean module
    mean_module u_mean (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_ready(mean_ready),
        .a_in(a_in),
        .mean(mean_val),
        .mean_sq(mean_sq),
        .out_valid(mean_done),
        .out_ready(out_ready)
    );

    // Var module
    var_module u_var (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_ready(var_ready),
        .a_in(a_in),
        .ex2(ex2),
        .out_valid(var_done),
        .out_ready(out_ready)
    );

    assign in_ready = mean_ready & var_ready;
    assign mean = mean_val;

    //-------------------------------------------
    // 내부에서 분산 계산: var = (ex2 - mean_sq) >> 8
    //-------------------------------------------
    wire [31:0] var_q16;

    // 빼기: 1-cycle delay
    c_addsub_sub32 sub_inst (
        .A(ex2),
        .B(mean_sq),
        .CLK(clk),
        .CE(1'b1),
        .S(var_q16)
    );

    // var_q16[23:8] → Q8.8 정규화 출력 저장
    reg [15:0] var_reg;
    reg        out_valid_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            var_reg       <= 16'd0;
            out_valid_reg <= 1'b0;
        end else begin
            var_reg       <= var_q16[23:8];
            out_valid_reg <= mean_done & var_done;
        end
    end

    assign var        = var_reg;
    assign out_valid  = out_valid_reg;

endmodule

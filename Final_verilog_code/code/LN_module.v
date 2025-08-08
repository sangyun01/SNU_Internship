module LN_module (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              in_valid,
    output wire              in_ready,
    output wire              out_valid,
    input  wire              out_ready,
    input  wire [64*16-1:0]  a_in,
    output wire [64*16-1:0]  norm_out,
    
    output wire [15:0]       mu_out,
    output wire [15:0]       var_out,
    output wire [15:0]       inv_std_out,
    output wire [64*16-1:0]  diff_out_dbg   
);

    assign mu_out       = mu;
    assign var_out      = var;
    assign inv_std_out  = inv_std;
    assign diff_out_dbg = diff_out;
    // Stage 1: 입력 latch
    reg [64*16-1:0] x_reg;
    reg             x_valid;
    wire            mean_var_ready;

    assign in_ready = ~x_valid | mean_var_ready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_valid <= 1'b0;
        end else if (in_valid && in_ready) begin
            x_reg   <= a_in;
            x_valid <= 1'b1;
        end else if (mean_var_ready) begin
            x_valid <= 1'b0;
        end
    end

    // Stage 2: mean/var
    wire [15:0] mu, var;
    wire        mean_var_valid;
    wire        pwl_ready;

    mean_var u_meanvar (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(x_valid),
        .in_ready(mean_var_ready),  // 핸드셰이크 그대로 사용 가능
        .a_in(x_reg),
        .mean(mu),
        .var(var),
        .out_valid(mean_var_valid),
        .out_ready(pwl_ready)
    );

    // Stage 3: PWL (1 / sqrt(var))
    wire [15:0] inv_std;
    wire        pwl_valid;
    wire        sub_ready;

    PWL_module u_pwl (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(mean_var_valid),
        .in_ready(pwl_ready),
        .x_in(var),
        .y_out(inv_std),
        .out_valid(pwl_valid),
        .out_ready(sub_ready)
    );

    // Stage 4: Subtract (x_i - mu)
    wire [64*16-1:0] diff_out;
    wire             sub_valid;
    wire             mult_ready;

    normalize_sub_module u_sub (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(pwl_valid),
        .in_ready(sub_ready),
        .x_in(x_reg),
        .mu(mu),
        .diff_out(diff_out),
        .out_valid(sub_valid),
        .out_ready(mult_ready)
    );

    // Stage 5: Multiply (normalize)
    normalize_mult_module u_mult (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(sub_valid),
        .in_ready(mult_ready),
        .diff_in(diff_out),
        .inv_std(inv_std),
        .norm_out(norm_out),
        .out_valid(out_valid),
        .out_ready(out_ready)
    );
    
endmodule

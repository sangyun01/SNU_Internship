module mul #(
  parameter N = 16,               // 입력 비트수 (N = 16)
  parameter FRAC_BITS = 8         // 소수부 비트 수 (Q8.8 → 8)
)(
  input                     clk,
  input                     rst,
  input  signed [N-1:0]     a,
  input  signed [N-1:0]     b,
  output signed [N-1:0]     result
);

  // result of bit size : 2N bit
  reg signed [(2*N)-1:0] full_r;

  // using pipeline
  reg signed [N-1:0] result_r;

  always @(posedge clk) begin
    if (rst) begin
      full_r   <= 0;
      result_r <= 0;
    end else begin
      full_r   <= a * b;                             // 고정소수점 곱셈
      result_r <= full_r[(2*FRAC_BITS)+N-1:FRAC_BITS]; // Q(FRAC).(N-FRAC) 결과 추출
    end
  end

  assign result = result_r;

endmodule

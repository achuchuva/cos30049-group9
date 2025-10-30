import { render, screen } from '@testing-library/react';
import App from './App';

test('renders spam detection demo heading', () => {
  render(<App />);
  const heading = screen.getByText(/Spam Detection Demo/i);
  expect(heading).toBeInTheDocument();
});
